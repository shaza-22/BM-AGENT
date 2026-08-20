# Banque Misr Agentic Research Assistant — Browsing & Navigation

Two layers of the agent live here:

- **`browsing/`** — perception. `fetcher.py` (how the agent sees a page) and
  `extract_links.py` (how it sees where it can go next).
- **`agent/`** — navigation. `link_selector.py` (which link to follow, and why)
  and `navigator.py` (the loop from the homepage to the answering page).

Planning, content extraction, validation and synthesis sit above these and are
built separately.

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
  llm.py             LLMClient protocol, ClaudeLLMClient, FakeLLMClient
  link_selector.py   ranking, prompt, defensive parsing, select_next_link
  navigator.py       the navigation loop -> NavigationResult
  trail_log.py       JSON Lines step log for the frontend and evaluation
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
pytest                                  # 263 tests, no network, no API key
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
