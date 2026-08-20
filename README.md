# Banque Misr Agentic Research Assistant — Browsing Layer

The navigation/perception half of the agent: **`browsing/fetcher.py`** (how the
agent sees a page) and **`browsing/extract_links.py`** (how it sees where it can
go next). Planning, content extraction, validation and synthesis sit above this
layer and are built separately.

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
scripts/
  save_fixtures.py   one-off: snapshot live pages into fixtures/live/
fixtures/
  fixture_urls.txt   the pages to snapshot (data, not code)
  synthetic/         hand-written pages reproducing each site quirk
  live/              real snapshots, produced by save_fixtures.py
tests/               pytest suite, fully offline
```

## Install & run

```bash
pip install -r requirements.txt
pytest                                  # 145 tests, no network
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
