# Banque Misr Agentic Research Assistant — Browsing, Navigation & Interface

- **`browsing/`** — perception. `fetcher.py` (how the agent sees a page) and
  `extract_links.py` (how it sees where it can go next).
- **`agent/`** — navigation. `link_selector.py` (which link to follow, and why),
  `navigator.py` (the loop from the homepage to the answering page), and
  `session.py` (conversation memory, so follow-ups work).
- **`api/` + `frontend/`** — a thin HTTP layer over `navigate()` and a
  single-page UI that shows each hop and its reasoning as it happens.

Planning a task into sub-goals, validating a page, extracting fields and
synthesising an answer sit above these and are built separately. `validate_fn`
is still a stub here, and the API injects it, so that work drops in without
reshaping anything.

The agent gets one seed URL and a task, and navigates the live site itself.
There is no pre-built index and no crawler; every task starts fresh from the
homepage and follows links at runtime.

## Layout

```
browsing/
  config.py          every tunable: language policy, domain, delays, thresholds
  fetcher.py         Fetcher, fetch_page, robots_allowed, html_to_text, pdf_to_text
  extract_links.py   normalize_url, canonical_key, extract_links, link_label
  _demo.py           argument handling for the __main__ blocks (not agent API)
agent/
  config.py          seed URL, caps, model, ranking weights, WAF markers
  session.py         Turn/TurnSummary, SessionStore, follow-up resolution
  llm.py             LLMClient protocol, ClaudeLLMClient, FakeLLMClient
  link_selector.py   ranking, prompt, defensive parsing, select_next_link
  navigator.py       the navigation loop -> NavigationResult
  trail_log.py       JSON Lines step log for the frontend and evaluation
api/
  app.py             FastAPI routes and the SSE stream
  runner.py          task registry, worker pool, shared rate limiter
  events.py          SSE framing
  schemas.py         the wire format, in one place
frontend/
  index.html         the whole UI: no build step, no framework, no npm
scripts/
  save_fixtures.py   one-off: snapshot live pages into fixtures/live/
  live_navigate.py   watch one sub-goal navigate (demo / smoke check)
fixtures/
  fixture_urls.txt   the pages to snapshot (data, not code)
  synthetic/         hand-written pages reproducing each site quirk
  live/              real snapshots, produced by save_fixtures.py
tests/               pytest suite, fully offline
```

## Install & run

```bash
pip install -r requirements.txt
cp .env.example .env                    # then paste your key into it
pytest                                  # 487 tests, no network, no API key
python scripts/save_fixtures.py         # ONE-OFF, hits the live site
```

The two inspection blocks read saved pages and never touch the network. Both
accept a directory, an explicit file, or a glob pattern (handled internally,
since neither cmd.exe nor PowerShell expands one), and default to the synthetic
fixtures:

```bash
python -m browsing.extract_links                          # synthetic fixtures
python -m browsing.extract_links fixtures/live            # a directory
python -m browsing.extract_links fixtures/live/index.html # one file
python -m browsing.fetcher "fixtures/live/*.html"         # a glob
```

When a `manifest.json` sits beside the fixture, each page is parsed against the
URL it was actually saved from, so relative hrefs resolve the way they did on
the live site.

### Choosing a model provider

`agent/config.py` selects the provider; `agent.llm.make_llm_client()` builds it:

```python
PROVIDER = "gemini"          # "gemini" | "claude"
GEMINI_MODEL = "gemini-2.5-flash"
CLAUDE_MODEL = "claude-opus-5"
```

Both implement the same `LLMClient` protocol, so nothing above `agent/llm.py`
knows which one is in use. Each SDK is a lazy import, so only the provider you
actually use needs installing — `google-genai` for Gemini, `anthropic` for
Claude. Override per run with `--provider` / `--model` on `live_navigate.py`.

**API keys.** Read from the environment, falling back to a `.env` file at the
project root (`python-dotenv`; an exported variable always wins). `.env` is
gitignored — start from `.env.example`. Keys are resolved at connect time and
handed straight to the SDK: never stored on a client object, never in a repr,
never in a log line or an error message. Error text from either SDK is scrubbed
of key-shaped strings before it is raised, and the raw SDK exception is
deliberately not chained, so it cannot resurface in a traceback. A missing key
produces a message naming the *variable*, not a raw SDK failure:

```
no Gemini API key found: set GEMINI_API_KEY in the environment, or put
GEMINI_API_KEY=<your key> in /path/to/BM-AGENT/.env (that file is gitignored)
```

### Transient failures

Free-tier endpoints return `503 UNAVAILABLE` regularly, and the selector is
called once per hop, so one unlucky call used to end a whole run at zero hops.
Both clients now retry transient failures with exponential backoff, mirroring
what `browsing/fetcher.py` already does for HTTP:

```python
LLM_MAX_ATTEMPTS = 4        # 4 attempts: waits of 2s, 4s, 8s
LLM_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
LLM_PERMANENT_STATUS = {400, 401, 403, 404, 405, 422}
```

Permanent answers — a bad key, an unknown model — fail on the first attempt,
because retrying only delays a clear message. Status-less failures are retried
only when they look like a connection or overload problem; a `TypeError` in our
own request is not. Google's `UNAVAILABLE`-style status names are recognised
alongside the numeric codes. Every retry is logged, so a slow hop explains
itself:

```
WARNING agent.llm: Gemini call failed (status=503, attempt 1/4): … -- retrying in 2.0s
```

Worst case a hop now takes ~14s longer before giving up rather than failing
instantly — the right trade for a live demo.

### Where a run's time goes

There is **no client-side rate limiter for the model** — nothing self-paces
against a requests-per-minute quota. Only three things ever sleep:

| | Where | Cost |
|---|---|---|
| politeness pacing | `browsing/fetcher.py` `RateLimiter` | 1–2s before each request to a host, first one free |
| HTTP retry backoff | `browsing/fetcher.py` | 2s, one retry per page |
| model retry backoff | `agent/llm.py` | 2s + 4s + 8s = 14s worst case per call |

So a 3-page, 3-call run spends 2–4s on pacing. If it takes minutes, the time is
in the model, the site, or retries — and each is now reported separately.

Every fetch line splits the total, including the work done on the bytes after
they arrive:

```
fetch url=… wait_ms=1400 retry_ms=0 req_ms=6100 parse_ms=95 pdf_ms=0 ms=7600
rate-limit wait host=www.banquemisr.com slept=1.42s (politeness delay …; not a retry)
```

and `NavigationResult.stats["timing"]` breaks down the whole run, which
`live_navigate.py` prints:

```
time     :
  model calls               0.8s  (2 calls)
  model retry backoff       2.0s  (1 retries)
  page requests             0.0s  (2 pages)
  politeness pacing         0.0s  (1-2s between requests to the same host)
  page retry backoff        0.0s
  robots.txt                0.0s
  pdf extraction            0.0s
  html parsing              0.0s
  link extraction           0.0s
  unaccounted               0.0s  (validation, everything else)
```

If `model retry backoff` dominates, the endpoint is failing and
`LLM_MAX_ATTEMPTS` is the knob. Only if `politeness pacing` dominates is
`DELAY_RANGE_S` worth touching, and on a WAF-protected site it is the last
thing to cut. In practice the model dominates — see below.

### Reasoning depth is the dominant cost of a hop

Measured on a two-hop run: **35.6s of 38.7s inside the model**, and at identical
prompt sizes one call took 3.5s while another took 32.8s. Link selection is a
model picking one entry from a list of labels — it does not need to reason at
length, and the difference is most of the run.

Each provider has a knob, and both are in `agent/config.py`:

```python
GEMINI_THINKING_LEVEL = "low"   # what gemini-3.x takes
GEMINI_THINKING_BUDGET = None   # what older models take; 0 turns reasoning off
CLAUDE_EFFORT = "low"           # the Claude counterpart, already at the cheap setting
```

**`thinking_budget` is rejected by gemini-3.x** — with a bare 400, not a message
that names it — so `thinking_level` is the working knob there. The budget is
kept for older models. Measured on 3.6-flash: `thinking_level="low"` took a
32.8s call down to **1.6s**, with choice quality and the arrival signal both
intact. Whichever knob is set is sent; if the API refuses it the client drops it
and carries on (see below), so an unsupported setting costs the speedup and
never the call.

### When the provider refuses a request

Structured output and the thinking configuration are best-effort. Google refuses
an unsupported one with a bare `400 INVALID_ARGUMENT: Request contains an
invalid argument.` that names no field, so matching on the message cannot work.
The handling is structural instead: **on any 400 where an optional setting was
sent, they are dropped one at a time** — `LLM_OPTIONAL_SETTING_DROP_ORDER`,
thinking first because it is the likelier culprit and the cheaper loss — and the
call retried after each.

A setting is only disabled for the rest of the run **if dropping it actually made
the call succeed**. A request that was malformed for some unrelated reason
restores everything and raises the original error, so a bad request never
quietly degrades the run's quality. Shedding is not a retry: it does not count
towards the retry budget and takes no backoff.

### Quotas

A 429 is inspected structurally, through the `google.rpc.QuotaFailure` detail
rather than the message:

- **Per-day quotas fail immediately.** `GenerateRequestsPerDayPerProjectPerModel`
  does not refill in seconds, so backing off inside a run spends time to fail
  anyway — it cost 14s across four attempts before this. The error names the
  quota and says plainly that retrying will not help.
- **Per-minute quotas are retried**, honouring `RetryInfo.retryDelay` when the
  server sends one, capped at `LLM_RETRY_MAX_BACKOFF_S`. The log says which:

```
… -- retrying in 27.0s (server-requested)
… -- retrying in 2.0s (exponential)
```

Claude's counterpart is `effort`, not a thinking switch. *Disabling* thinking on
that model is deliberately not offered: with thinking off it can write a tool
call into visible text or leak reasoning tags, so lowering effort is the
supported way to spend less.

Every call reports itself, so before/after is measurable without a stopwatch:

```
llm call provider=Gemini model=gemini-3.6-flash ms=3421 thinking=off schema=on
llm call provider=Claude model=claude-opus-5   ms=2100 effort=low  schema=on
```

### PDF extraction is the expensive part of a hop

Measured on an 84-page, 6.9MB tariff (~129k characters):

| | time | keeps |
|---|---|---|
| pdfplumber, text + tables | 24.4s | column structure |
| pdfplumber, text only | 15.3s | prose only |
| pypdf, text only | **6.3s** | all the content, flat |

On a *7-page* document the same comparison is 1.8s / 1.2s / 2.2s — pypdf is
slower there. Neither extractor is simply better; the document's size decides,
so `pdf_to_text` picks by page count:

```python
PDF_FIDELITY_MAX_PAGES = 25   # above this, use the fast flat-text path
PDF_EXTRACT_TABLES = True     # table rendering is 37% of pdfplumber's cost
```

Below the threshold, pdfplumber keeps the column structure that makes a fee row
readable — the reason PDFs are worth following at all. Above it, the run
switches to pypdf: **no content is lost, only column alignment**, and a document
that large is past the point where its text would be handed to a model whole.
Counting the pages first costs ~45ms. The choice is logged, so a slow or flat
extraction is never a mystery:

```
pdf has 84 pages (over the 25-page fidelity limit) -- using the fast flat-text
extractor; table columns will not be preserved
```

Measured end to end on that document: **22.4s → 5.3s**, and the extracted text
halves too (285k → 140k characters), because table rendering deliberately
overlaps the flat pass.

If you need full table fidelity on a long document, raise
`PDF_FIDELITY_MAX_PAGES` and accept the time.

### Provider differences worth knowing

The two providers bind structured output differently, and **neither guarantees a
reply the selector can use as-is**:

| | Claude | Gemini |
|---|---|---|
| structured output | `output_config={"format": {"type": "json_schema", …}}` | `response_json_schema` + `response_mime_type` |
| schema dict | the same `SELECTION_SCHEMA` for both, unmodified | ditto |
| if the schema is refused | the request fails | falls back to plain text and keeps going |
| reasoning depth | `effort="low"` (adaptive thinking) | model default |

Gemini's binding was validated against the SDK's own type checking, not against
a live call, so treat it as unconfirmed until you have run it once. If the API
refuses the schema, `GeminiLLMClient` logs a warning, disables structured output
for the rest of the run, and continues on plain text — which is exactly when
`link_selector.parse_selection` earns its keep. That parser strips markdown
fences, recovers JSON embedded in prose, coerces string and float indices,
rejects out-of-range and boolean choices, and turns anything else into an
explicit "no candidate" with the error in `reasoning`. It is tested against both
clients with the same set of malformed replies, so switching provider cannot
silently change navigation behaviour.

One case is deliberately *not* treated as a parse failure on either provider: an
empty reply raises `LLMError` and ends the run with `status="error"`. Parsing it
as "no candidate fits" would read as "the site does not cover this", which is a
different and wrong conclusion.

Optional browser rendering, needed for roughly 3% of pages:

```bash
pip install playwright && playwright install chromium
```

If Playwright is absent the fetcher logs a warning and returns the `requests`
result, so the project runs for someone without browsers installed.

> On some Debian-based images `pdfplumber` fails to import with
> `ModuleNotFoundError: _cffi_backend`, caused by the distro `cryptography`
> package. `pip install --upgrade cffi` fixes it.

## Using it

```python
from browsing import Fetcher, extract_links

fetcher = Fetcher()                      # one per task; discard when the task ends
page = fetcher.fetch("https://www.banquemisr.com/")
if page["ok"]:
    for link in extract_links(page["raw_html"], page["url"]):
        ...   # {"label", "url", "key", "source", "is_pdf"} -> the planner picks one
print(fetcher.stats)                     # fetches, cache hits, escalations, errors
```

`fetch` never raises. A failed hop returns `ok: False` with an `error`, so the
agent loop can try a different link instead of losing the task.

## Navigating

```python
from agent import Navigator
from agent.llm import make_llm_client

navigator = Navigator(make_llm_client())          # one per task; provider from config
result = navigator.navigate("find the annual fee of the classic credit card")

print(result.status)          # resolved | exhausted | no_candidates | blocked | error
print(result.sources)         # every URL actually fetched -- the evidence
for step in result.trail:
    print(step.hop, step.label, step.reasoning)
```

Watch a run:

```bash
python scripts/live_navigate.py "Where are the fee schedules?" --offline
python scripts/live_navigate.py "Find the classic credit card" --resolve-on "annual fee"
```

`--offline` replays `fixtures/live/` through the real `Fetcher`, so you can tune
the prompt against real pages without touching the site. Reach for it by
default: the site is behind an F5 WAF that bans on burst traffic.

`validate_fn(sub_goal, page) -> {"resolved", "extracted", "reason"}` is the seam
for the validation layer. It receives the whole `PageDict` (`raw_html` for
tables, `text` for prose, `url` for attribution) and defaults to a stub that
never resolves, so the loop is exercisable before the real validator exists.

### Cost and pacing per hop

Measured on the real homepage: the system prompt plus 50 candidates is ~5.8k
characters, roughly **1,450–1,700 input tokens**. With ~400 output tokens
(reasoning plus adaptive thinking at `effort="low"`) that is about **$0.02 a
hop**, so **~$0.10 for a sub-goal** that runs to the 5-hop cap.

Prompt caching is deliberately not used: the stable prefix is ~250 tokens and
the minimum cacheable prefix is ~1024, so a breakpoint would silently never hit.

Wall clock matters more than cost for a demo — 5 hops of (1–2s politeness delay
+ fetch + one model call) is **20–35s per sub-goal**.

## The web interface

```bash
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000>. Bound to localhost deliberately: the domain
allow-list means this cannot be pointed at another host, but an exposed
endpoint would still let anyone spend the day's model quota and drive traffic
at a WAF-protected bank from your address.

| endpoint | |
|---|---|
| `POST /api/sessions` | start a conversation |
| `POST /api/tasks` | `{task, session_id?}` → `{task_id, session_id, state}`, returns at once |
| `GET /api/tasks/{id}/stream` | SSE: `resolved` → `hop`… → `done` \| `error` |
| `GET /api/tasks/{id}` | the same events accumulated — the poll fallback |
| `GET /api/health` | liveness, provider, and whether a key is configured (never the key) |

A run takes 5–30s, so progress streams rather than the request hanging. `done`
or `error` **always** terminates the stream, including on an unhandled
exception, so a client can never wait forever. If the stream never connects or
drops, the UI falls back to polling; the status endpoint accumulates everything
the stream emitted, so replaying it is safe.

Everything is in memory. **If the server restarts, every task id, session and
in-flight run is lost** — a poll for a pre-restart id returns 404, which the UI
reports as a lost run rather than a missing one. A client disconnecting does
not cancel a run; it finishes and waits in the registry until its TTL.

### Concurrency against a WAF

Two mechanisms, because neither is sufficient alone:

- **One `RateLimiter` shared by every run in the process.** It keys per host
  under a lock, so however many runs are active, requests to banquemisr.com
  stay at one per 1–2s. A concurrency cap alone would not do this — two runs at
  one fetch per 1.5s each is one fetch per 0.75s, twice the intended rate.
- **A worker pool of two.** With a shared limiter and no cap, ten concurrent
  runs interleave their fetches and all ten take ten times as long. With a cap,
  the third request queues and is told its position.

Each run still gets its own `Fetcher`: the per-run cache and visited-set belong
to one task. Runs are blocking, so they execute on the pool rather than the
event loop, and hand events back through `call_soon_threadsafe` on the loop
that owns the stream.

## Session memory and follow-ups

`POST /api/tasks` with a `session_id` sends the request through a resolver
first: an LLM call that rewrites "and the fees for that?" into a standalone
sub-goal using the earlier turns. Rules-based pronoun matching was rejected —
it would be brittle and the patterns would encode the site's vocabulary, which
is the coupling this project avoids everywhere else.

The resolved sub-goal is always surfaced ("Interpreting as: …"), which is the
real guard: a wrong rewrite is visible rather than silent. A resolver that
fails, returns nothing, or answers the question instead of rewriting it falls
back to the raw request with a logged warning, and never blocks a run.

A turn stores a **`TurnSummary`**, not a `NavigationResult`: statuses, source
URLs, and each hop's label and reasoning. A `NavigationResult` carries a
`PageDict` with up to 140k characters of text and 300KB of raw HTML, and
keeping those per turn would both leak memory and dump hundreds of KB into
anything that serialises a turn. Sessions are bounded by turns, age and count.

### Carrying prior-visited URLs into a follow-up

A follow-up about a page found earlier could skip re-walking the site. There
are four ways to do that and only one is safe:

| approach | verdict |
|---|---|
| seed the frontier with prior URLs | **breaks the central claim** — the agent hops to a page it never discovered this run; that is a small pre-built index |
| pass them as `exclude_urls` | actively wrong — "and the fees for that?" usually needs to re-read the page the last turn ended on |
| cache pages across turns | breaks "every task starts live" outright, and serves stale content |
| **a ranking bonus only** | safe — the link must still be discovered this run from the seed, the model still chooses, the caps still bound the run |

Only the fourth is implemented, and it **ships disabled**
(`FOLLOW_UP_PATH_BONUS = 0.0`). It is safe, but it buys roughly ten seconds of
re-walking at the cost of complicating the clearest sentence in the project —
*no pre-built index, every task navigates live from the homepage* — and it
would pull a follow-up that is really a topic switch back toward the previous
topic. Set the constant above zero to enable it; it is gated on `used_context`
so a self-contained request is never affected.

## Bilingual operation

The site is fully bilingual and some content exists in one language only. A run
carries a language, chosen per task rather than per process, so two tasks in
different languages can share one server.

**Detection costs nothing.** The task's script decides: Arabic letters against
Latin letters, digits and punctuation ignored, Arabic winning above
`ARABIC_DETECTION_THRESHOLD` (0.30). Below half on purpose — Latin brand names
inside Arabic questions are common ("BM Wallet ازاى" is 0.33 and is Arabic),
while Arabic words inside English questions are rare. An explicit `language` on
the API request or `--language` on the CLI overrides detection. No model
request is spent on something Unicode already answers.

**`sc_lang` is a meaningful parameter, not tracking.** This was the load-bearing
fix. Verified on the saved pages: the Arabic site is *not* a separate path tree,
it is the same paths carrying `?sc_lang=ar-EG`, and every English page links to
its own Arabic twin that way. `sc_lang` used to be stripped as a Sitecore
tracking parameter, which meant an Arabic URL normalised into the English page —
so an Arabic run fetched English content believing it was Arabic — and both
languages collapsed onto one canonical key, so visiting one marked the other
visited. It is now stripped only when it names the site's default language.

**A link's language comes from its URL marker, falling back to its label's
script.** The site's markers are inconsistent; an unmarked URL under an Arabic
label is an Arabic page, and the label is free to read.

### Cross-language policy: permissive with penalty

Other-language links stay in the frontier, ranked below the run's own by
`PENALTY_OTHER_LANGUAGE`. `CROSS_LANGUAGE_POLICY = "strict"` drops them instead.

Permissive is the default because **strict fails silently and totally**: content
that exists in one language only becomes unreachable and the agent reports "not
found on the Banque Misr website" — a false negative, against the criterion the
assignment grades as missing-information handling. Claiming something is
unavailable when it is available is the worst answer there. Permissive fails
visibly and partially instead: one other-language URL in a source list, plainly
shown in the trail. It also matches the site's own structure, where a language
switcher on every page makes crossing a single hop by design, and it is the same
shape as every other decision here — nav links, alias links and passed-over
links are all penalised, never dropped.

### Answering in the language asked

The model is instructed to write its `reasoning` in the task's language; the
JSON field names and the `outcome` enum stay English, so parsing is unaffected.
The sentences the *code* writes — running out of links, hitting a cap, failing
to parse a reply — live in `agent/messages.py` with an Arabic variant each. A
run that explains its successes in Arabic and its failures in English would be
worse than one that is consistent, and failures are where a user most needs to
understand what happened. The UI uses `dir="auto"` on labels, reasoning and the
resolved sub-goal, which is per-string by definition, so one run can hold both
scripts without flipping the page; URLs and badges stay LTR.

## Demoing without spending quota

The free tier allows about twenty model requests a day and a sub-goal costs
three or four, so a rehearsal plus a presentation can exceed it.

```bash
BM_RECORD_DIR=recordings uvicorn api.app:app     # save each finished run
BM_REPLAY_DIR=recordings uvicorn api.app:app     # serve saved runs, no model calls
```

A replayed run is announced as one — `replayed: true` on its `resolved` event
and a badge in the UI. Presenting a recording as a live run would undo the
honesty the rest of this project is built on.

## Design decisions

**Select on `[href]`, not `a[href]`.** The main navigation is Vue components
(`<b-dropdown-item href="...">`), not anchors. `find_all("a", href=True)`
returns zero main-nav links and strands the agent on the homepage. The footer
does use real anchors, so both forms coexist.

**Labels can never be empty.** Homepage tiles are image-only, and a link with no
label is invisible to the LLM choosing the next hop. Fallback chain: element
text → nested heading → `img alt` → `aria-label`/`title` → URL slug. The slug
works unusually well here because Sitecore paths read like prose
(`/Pages/Credit%20Cards%20List` → "Credit Cards List").

**Normalisation and identity are separate jobs.** `normalize_url` returns a
*fetchable* URL and preserves path case, because Sitecore paths can be
case-sensitive and a lowercased URL may 404. `canonical_key` returns a
lowercased, separator-folded *identity* used for the visited-set and the cache,
so `Retail%20Banking`, `retail-banking` and `Retail_Banking` collapse to one
entry. Both are logged on every fetch, so dedupe bugs show up in the step log.

**`csrt` is stripped before anything else.** It is a per-request anti-CSRF token:
the same page returns a different one every time. Left in, nothing ever dedupes
and the agent re-fetches one page until its budget is gone. `sc_site`, `sc_mode`
and `sc_itemid` are stripped for the same reason.

**Language is filtered before parameters are stripped.** `sc_lang` is itself a
stripped parameter, so the other order erases the only evidence that a URL is
Arabic. `LANGUAGE` in `config.py` controls this; set it to `None` to disable.

**Footer links are tagged, never dropped.** The fees hub is reachable only from
the footer. `source` (`nav`/`body`/`footer`) is metadata for debugging and
prompt context; when detection is unsure it returns `"body"`, because a
mislabelled link is an annoyance and a dropped link is an unanswerable task.

**Requests first, browser only on evidence.** A survey found ~1931 pages fine
with plain `requests` against ~65 needing a browser. Escalation fires on
`C` empty body → `B` text under 200 chars → `D` SPA shell marker with thin text
→ `A` fewer than 5 links, and the reason code is logged. The rendered result is
kept only if it actually has more text or more links, so `render_mode` always
names what was returned and `stats` reports how often escalation helped.

**PDFs are identified by Content-Type and magic bytes, not by path.** Sitecore
serves PDFs from `/-/media/...ashx`, but that same prefix serves images. The
`is_pdf` field on a link is a *hint* for the planner; `fetch` decides the truth.

**Domain matching is exact-or-subdomain.** `host.endswith("banquemisr.com")`
also accepts `evil-banquemisr.com`; `digital.banquemisr.com` is allowed by the
subdomain branch.

**The cache is per-run and in memory only.** It exists so revisiting a page
*within one task* does not re-fetch it. It is never written to disk and never
shared between tasks — every task re-reads the live site. All mutable state
lives on the `Fetcher` instance, so tasks can run concurrently. One caveat: two
instances do not share a politeness clock, so N concurrent tasks hit the host at
N times the rate. Pass one shared `RateLimiter` to every `Fetcher` if that
matters.

**Frontier search, not a strict tree-walk.** Candidates accumulate across the
whole run, so hop 4 can follow a link first seen on hop 1. The site repeats its
entire navigation on every page — 60 of the 173 distinct links across the saved
pages appear on all of them — so the link that answers a sub-goal is often one
that was already on screen two hops ago. With a five-hop budget there is no room
to re-walk a tree to reach it. Every `TrailStep` records `from_url` and
`discovered_at_hop`, so a lateral jump reads plainly in the log; a depth-first
run is just the case where `discovered_at_hop == hop - 1` throughout.

**Ranking is structural and never reads the sub-goal.** Links are scored on
where they were found, how long ago, which page region, and how often they have
already been offered — never on matching sub-goal words against labels. A
lexical filter would quietly become the agent and would fail on any sub-goal
phrased differently from the site's own vocabulary. The model does the semantic
work; ranking only decides what fits in the prompt.
`test_ranking_is_independent_of_the_sub_goal` pins this down.

**Passed-over links decay, they are not dropped.** A link the model saw and did
not choose was not rejected — only out-ranked — and it is the natural next try
when the first choice dead-ends. Same for nav links after the first hop: they
are penalised because they are identical on every page, but kept because they
are the only route into a different section after a wrong turn.

**No page region may be crowded out.** Measured: the homepage has 40 body links
against 11 footer ones, so a straight top-N cut offered zero footer links —
putting the footer-only fees hub and sitemap out of reach on the first hop.
Regions now get reserved slots, filled round-robin and capped at half the list
so they can never starve the ranking itself.

**The model picks a candidate by index, not by URL.** A model asked for a URL
will occasionally invent one or reword an encoded path. An index either names a
link the agent actually found or is out of range, which is detectable and
becomes "no candidate" instead of a fetch to a made-up address.

**"None of these" is a first-class outcome.** When the site does not cover a
sub-goal, the correct answer is to say so. That returns `no_candidates` with the
model's reasoning, not a low-confidence guess. Self-reported confidence is
logged but not gated on by default — it is weakly calibrated — with
`MIN_CONFIDENCE` available if you want a threshold.

**Arrival and resolution are different claims, and only one component makes
each.** "No link to follow" used to mean both "nothing here is relevant" and
"we have arrived, stop walking", so a successful walk reported as a failure. The
selector now returns `follow` / `arrived` / `none`, and the navigator maps
`arrived` to its own status.

The precedence is one-way and absolute: **only `validate_fn` can produce
`resolved`.** The selector sees link labels and URLs, never page content, so
`arrived` is a statement about the *route* — "there is nowhere better to go from
here" — and can never assert that the sub-goal is answered. An `arrived` result
returns the page it reached (so a caller can extract from it) with `extracted`
left `None`.

Anything unrecognised — an omitted field, an unknown value, an unparseable
reply — defaults to `none`, never `arrived`: mistaking a confused reply for a
successful arrival would report a failed run as a finished one. A page with no
unvisited links left is likewise not assumed to be an arrival, because the
selector is never consulted there and reporting one would be a guess.

Once a real validator is installed, `arrived` should become rare — and each
occurrence is worth inspecting, because it means the navigator believed it had
arrived and the validator disagreed. Those are the highest-value rows in the
evaluation pipeline, and today they are indistinguishable from genuine dead
ends.

**A WAF block page aborts the whole run.** The F5 WAF answers HTTP 200 with an
"Access Denied" page, so `fetch_page` reports `ok=True` and the agent would
otherwise parse the refusal as content and could cite it. It is detected
structurally (marker text plus a near-empty link list — a real page carries 70+)
and returns `status="blocked"`, because continuing to hit a WAF that has already
flagged you is how a run becomes a ban.

**Nothing here is topic-specific.** No product names, no category keywords, no
per-topic branching, no hardcoded page URLs. The only site-specific facts are
structural (the domain, Sitecore's parameter names, the language markers) and
all live in `config.py`. The fixture URL list is a data file for the same
reason.

## Testing

The suite is fully offline: HTTP is served by a fake session, HTML comes from
inline strings and `fixtures/synthetic/`, and the PDF path runs against a real
(hand-built) PDF. Fixture filenames differ by directory on purpose:
`fixtures/synthetic/` is hand-written and named descriptively
(`homepage.html`), while `fixtures/live/` is named from each page's canonical
key by `save_fixtures.py` (the homepage becomes `index.html`). Nothing depends
on a particular filename -- directories are scanned, not looked up. `fixtures/synthetic/` reproduces each documented site quirk —
the Vue nav, image-only tiles, heading+description links, `csrt` duplicates,
Arabic markers, footer-only routes, and an SPA shell.

Those synthetic pages are small, so they trip the escalation thresholds by
design; that is what makes the heuristic visible when you run
`python -m browsing.fetcher`.

## Known gaps

- The paths in `fixtures/fixture_urls.txt` were traced by hand and may be
  stale. `save_fixtures.py` records any that 404 in `manifest.json` and
  continues rather than failing the run — update the list, not the script.
- `robots.txt` has not been checked against the live site from this
  environment. `robots_allowed` fails *open* on a fetch error or 404 and fails
  *closed* on an explicit `Disallow`, logging loudly either way. Confirm what
  the live file actually says before relying on it.
- Playwright uses the synchronous API. An async agent loop would need the async
  variant; it is isolated in one method.
- The `/en/X-Y` ↔ `/Home/X%20Y` equivalence (`alias_key`) is a heuristic. It
  holds for the 18 duplicate pairs found in the saved pages, but nothing
  guarantees the two forms always serve identical content, so it only ever
  deprioritises and warns — it never drops a link or keys the cache.
- Per-hop token figures are character-based estimates, not `count_tokens`
  measurements, and the per-hop cost quoted above is Claude pricing. Re-measure
  before quoting either as a cost, and note that Gemini's free tier has request
  quotas rather than per-token billing.
- Neither provider has been exercised against its live API from this
  environment — both paths are tested against SDK-shaped stubs only. The first
  real call is worth watching with `--verbose`.
