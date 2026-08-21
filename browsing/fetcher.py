"""
Page retrieval for the Banque Misr agent.

What it does
    Fetches one URL and returns everything the rest of the agent needs to reason
    about it: the raw HTML (kept verbatim for downstream table extraction), a
    cleaned readable text rendering, and enough metadata to log and audit the
    step. Handles HTML and PDF, escalates to a headless browser only when a
    page looks under-rendered, and never raises on a bad fetch.

Inputs
    ``fetch_page(url, session=None)`` -- an absolute URL on the allowed domain.
    ``Fetcher(...)`` -- optional injected ``requests.Session``, robots policy,
    rate limiter and render policy.

Outputs
    A ``PageDict``::

        {"url", "requested_url", "status", "ok", "content_type",
         "raw_html", "text", "fetched_at", "render_mode", "error"}

    ``robots_allowed(url)`` -> bool.
    ``html_to_text(raw_html)`` -> readable text with table structure preserved.

Why it is needed
    ``fetch_page`` is the only way the agent perceives anything. Every claim in
    a final answer has to trace back to a page pulled through here during the
    current run, so this module is also where the audit trail starts -- one
    structured log line per fetch, carrying the canonical key so dedupe
    behaviour can be debugged after the fact.

Caching policy (deliberate, and load-bearing for the project's premise)
    The cache is an in-memory dict on a ``Fetcher`` instance, keyed by
    canonical URL. It exists so that revisiting a page *within one task* does
    not re-fetch it. It is never written to disk and never shared between
    tasks: every new task constructs a new ``Fetcher`` and therefore starts
    from the live site with an empty cache. There is no pre-built index here
    and there must not be one.

Concurrency
    All mutable state (cache, robots cache, politeness clock, counters) lives
    on the instance, so two tasks running at once share nothing. One caveat:
    two instances also do not share a politeness clock, so N concurrent tasks
    hit the host at N times the intended rate. Pass a single shared
    :class:`RateLimiter` to every ``Fetcher`` if that matters.
"""

from __future__ import annotations

import io
import logging
import random
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4.element import Comment, NavigableString, Tag

from browsing import config
from browsing.extract_links import canonical_key, extract_links, make_soup, normalize_url

logger = logging.getLogger(__name__)

ContentKind = Literal["html", "pdf", "other"]
RenderMode = Literal["requests", "playwright"]


@dataclass
class FetchTiming:
    """Where the wall-clock of one fetch went.

    Kept apart because they have different causes and different fixes: a
    politeness wait is our own pacing, a retry wait is the site failing,
    request time is the site being slow, and the last three are work this code
    does after the bytes arrive -- which on a large PDF dominates everything
    else.
    """

    wait_s: float = 0.0
    retry_wait_s: float = 0.0
    request_s: float = 0.0
    pdf_text_s: float = 0.0
    html_parse_s: float = 0.0
    link_extract_s: float = 0.0

    @property
    def total_s(self) -> float:
        return (
            self.wait_s + self.retry_wait_s + self.request_s
            + self.pdf_text_s + self.html_parse_s + self.link_extract_s
        )


class PageDict(TypedDict):
    """One fetched page. Always returned -- failures come back as ok=False."""

    url: str
    requested_url: str
    status: int
    ok: bool
    content_type: ContentKind
    raw_html: str | None
    text: str | None
    fetched_at: str
    render_mode: RenderMode
    error: str | None


# --------------------------------------------------------------------------
# Text rendering
# --------------------------------------------------------------------------
_INVISIBLE = str.maketrans(config.INVISIBLE_TRANSLATION)


def _clean_inline(value: str) -> str:
    return re.sub(r"[^\S\n]+", " ", (value or "").translate(_INVISIBLE)).strip()


def _render_rows(rows: list[list[str | None]]) -> str:
    """Render table rows as delimited lines.

    Cells are joined with " | " and rows with newlines. Without this a fee
    table collapses into one unbroken run of numbers and labels, and neither
    the LLM nor a human can tell which figure belongs to which row -- which is
    exactly the content the agent is most often asked to report.
    """
    lines: list[str] = []
    for row in rows:
        cells = [_clean_inline(cell or "") for cell in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _render_table(table: Tag) -> str:
    rows: list[list[str | None]] = []
    header_width = 0
    for index, row in enumerate(table.find_all("tr")):
        cells = row.find_all(["th", "td"])
        rows.append([cell.get_text(" ", strip=True) for cell in cells])
        if index == 0 and cells and all(cell.name == "th" for cell in cells):
            header_width = len(cells)

    body = _render_rows(rows)
    if not body:
        return ""
    if header_width:
        # A separator line after a header row keeps the header visibly attached
        # to its columns once the text is flattened into a prompt.
        first, _, rest = body.partition("\n")
        body = "\n".join([first, " | ".join(["---"] * header_width), rest]).rstrip("\n")
    return "\n" + body + "\n"


def html_to_text(raw_html: str) -> str:
    """Convert HTML to readable plain text, preserving table structure.

    Scripts, styles, noscript blocks and comments are removed; tables are
    rendered row-per-line with pipe-delimited cells; whitespace is collapsed
    without merging separate text nodes into one line.
    """
    if not raw_html:
        return ""

    soup = make_soup(raw_html)
    for tag in soup(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    # Reversed document order renders innermost tables first, so an outer
    # table picks up its nested table's already-formatted text.
    for table in reversed(soup.find_all("table")):
        table.replace_with(NavigableString(_render_table(table)))

    for br in soup.find_all("br"):
        br.replace_with(NavigableString("\n"))

    lines = []
    for line in soup.get_text("\n").splitlines():
        cleaned = _clean_inline(line)
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def _decode(content: bytes, content_type_header: str) -> str:
    """Decode response bytes without trusting requests' ISO-8859-1 default.

    ``requests`` falls back to ISO-8859-1 for text/* responses that omit a
    charset, which mojibakes this site's UTF-8 content (Arabic text and
    typographic punctuation both come through wrong). Header charset, then the
    document's own meta charset, then UTF-8.
    """
    for candidate in _charset_candidates(content, content_type_header):
        try:
            return content.decode(candidate, errors="replace")
        except (LookupError, UnicodeDecodeError):
            continue
    return content.decode("utf-8", errors="replace")


def _charset_candidates(content: bytes, content_type_header: str) -> list[str]:
    candidates = []
    match = re.search(r"charset=[\"']?([\w:.-]+)", content_type_header or "", re.I)
    if match:
        candidates.append(match.group(1))
    head = content[:4096].decode("ascii", errors="replace")
    match = re.search(r"charset=[\"']?([\w:.-]+)", head, re.I)
    if match:
        candidates.append(match.group(1))
    candidates.append("utf-8")
    return candidates


# --------------------------------------------------------------------------
# PDF text
# --------------------------------------------------------------------------
def _page_count(content: bytes) -> int | None:
    """Pages in a PDF, or None if it cannot be read cheaply.

    Costs ~45ms on a 7MB document -- only the cross-reference table is parsed --
    which is worth paying to pick the right extractor for the size.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    try:
        return len(PdfReader(io.BytesIO(content)).pages)
    except Exception:
        return None


def _pdfplumber_text(content: bytes, *, extract_tables: bool) -> tuple[str | None, str | None]:
    """High-fidelity extraction: keeps the column structure of tables."""
    try:
        import pdfplumber
    except ImportError:
        return None, "pdfplumber not installed"

    try:
        parts: list[str] = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    parts.append(page_text)
                if not extract_tables:
                    continue
                for table in page.extract_tables() or []:
                    rendered = _render_rows(table)
                    # Deliberately overlaps extract_text(): the flat pass loses
                    # column alignment, so the delimited copy is what makes a
                    # fee row readable downstream.
                    if rendered:
                        parts.append("[table]\n" + rendered)
        text = "\n".join(parts).strip()
        if text:
            return text, None
        return None, "pdfplumber extracted no text (scanned or image-only PDF?)"
    except Exception as exc:  # pdfplumber raises a wide variety of errors
        return None, f"pdfplumber failed: {exc}"


def _pypdf_text(content: bytes) -> tuple[str | None, str | None]:
    """Fast extraction: all of the content, none of the column structure."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return None, "pypdf not installed"
    try:
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        return (text, None) if text else (None, "pypdf extracted no text")
    except Exception as exc:
        return None, f"pypdf failed: {exc}"


def pdf_to_text(content: bytes) -> tuple[str | None, str | None]:
    """Extract text from PDF bytes. Returns ``(text, error)``.

    Which extractor runs depends on the document's size, because the right
    answer changes with it: pdfplumber keeps the table structure that makes a
    fee schedule readable, but its layout analysis dominates a long document
    (measured: 24.4s against pypdf's 6.3s on 84 pages, and the other way round
    on 7). Above :data:`config.PDF_FIDELITY_MAX_PAGES` the fast path is used --
    the full text, without column alignment -- and the choice is logged so a
    slow or flat extraction is never a mystery.
    """
    pages = _page_count(content)

    if pages is not None and pages > config.PDF_FIDELITY_MAX_PAGES:
        logger.info(
            "pdf has %d pages (over the %d-page fidelity limit) -- using the fast "
            "flat-text extractor; table columns will not be preserved",
            pages, config.PDF_FIDELITY_MAX_PAGES,
        )
        text, error = _pypdf_text(content)
        if text:
            return text, None
        logger.warning("fast pdf extraction failed (%s); falling back to pdfplumber", error)

    text, error = _pdfplumber_text(content, extract_tables=config.PDF_EXTRACT_TABLES)
    if text:
        return text, None

    fallback_text, fallback_error = _pypdf_text(content)
    if fallback_text:
        logger.info("pdf fallback to pypdf succeeded (%s)", error)
        return fallback_text, None
    return None, f"{error or ''} ; {fallback_error or ''}".strip(" ;")


# --------------------------------------------------------------------------
# Politeness
# --------------------------------------------------------------------------
class RateLimiter:
    """Per-host politeness clock, safe to share between concurrent Fetchers."""

    def __init__(self, delay_range: tuple[float, float] | None = None) -> None:
        self._delay_range = delay_range if delay_range is not None else config.DELAY_RANGE_S
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str, min_delay: float = 0.0) -> float:
        """Block until this host may be hit again. Returns seconds slept."""
        low, high = self._delay_range
        delay = max(random.uniform(low, high), min_delay)
        with self._lock:
            now = time.monotonic()
            last = self._last.get(host)
            sleep_for = 0.0 if last is None else max(0.0, delay - (now - last))
            # Reserve the slot before releasing the lock so a second thread
            # queues behind this request instead of racing it.
            self._last[host] = now + sleep_for
        if sleep_for > 0:
            # Logged before sleeping so a live run shows the pause as it
            # happens, and so a politeness wait is never mistaken for a retry
            # backoff or a slow server.
            logger.info(
                "rate-limit wait host=%s slept=%.2fs (politeness delay between "
                "requests to the same host; not a retry)",
                host, sleep_for,
            )
            time.sleep(sleep_for)
        return sleep_for


# --------------------------------------------------------------------------
# Fetcher
# --------------------------------------------------------------------------
class Fetcher:
    """Per-run page fetcher. Construct one per task; discard it when done."""

    def __init__(
        self,
        session: requests.Session | None = None,
        *,
        allow_playwright: bool = True,
        respect_robots: bool = True,
        language: str | None = None,
        rate_limiter: RateLimiter | None = None,
        delay_range: tuple[float, float] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._session = session if session is not None else requests.Session()
        headers = getattr(self._session, "headers", None)
        if headers is not None:
            headers.setdefault("User-Agent", config.USER_AGENT)
            if config.LANGUAGE:
                headers.setdefault("Accept-Language", config.LANGUAGE)
        self._allow_playwright = allow_playwright
        self._respect_robots = respect_robots
        # The run's language, not a process-wide constant, so two tasks in
        # different languages can share a process. None defers to the config.
        self.language = language
        self._rate_limiter = rate_limiter or RateLimiter(delay_range)
        self._timeout = timeout if timeout is not None else config.REQUEST_TIMEOUT_S
        self._max_retries = max_retries if max_retries is not None else config.MAX_RETRIES

        # Per-run state. Nothing here outlives the instance.
        self._cache: dict[str, PageDict] = {}
        self._robots: dict[str, tuple[RobotFileParser, float]] = {}
        self._stats = {
            "fetches": 0,
            "cache_hits": 0,
            "escalations": 0,
            "escalations_helped": 0,
            "pdfs": 0,
            "errors": 0,
            "robots_blocked": 0,
        }
        # Seconds, so a slow run can be attributed without a stopwatch.
        self._timing = {
            "rate_limit_wait_s": 0.0,
            "retry_wait_s": 0.0,
            "request_s": 0.0,
            "robots_s": 0.0,
            "pdf_text_s": 0.0,
            "html_parse_s": 0.0,
            "link_extract_s": 0.0,
        }

    # -- public ------------------------------------------------------------
    @property
    def stats(self) -> dict[str, Any]:
        """Counters for the run report (how often the browser was needed).

        ``rate_limit_wait_s`` is our own politeness pacing, ``retry_wait_s`` is
        time spent backing off after a failure, ``request_s`` is the site being
        slow, and ``pdf_text_s``/``html_parse_s``/``link_extract_s`` are this
        code's own work on the bytes. Keeping them apart is what makes a long
        run explainable.
        """
        # Millisecond resolution: rounding to hundredths reported a 6ms parse
        # as 0.0, which reads as "did not happen" rather than "was fast".
        return {**self._stats, **{k: round(v, 3) for k, v in self._timing.items()}}

    def clear_cache(self) -> None:
        self._cache.clear()

    def fetch(self, url: str) -> PageDict:
        """Fetch one URL. Never raises -- failures return ``ok=False``.

        The agent loop must be able to try a different link when a hop fails,
        so an exception here would cost the whole task.
        """
        started = time.perf_counter()
        normalized = normalize_url(url, url, language=self.language)
        if normalized is None:
            result = self._failure(
                url, url, 0, "url rejected by normalize_url (off-domain, asset, or wrong language)"
            )
            self._log(result, canonical_key(url) if url else "", 0, started, FetchTiming(), cache="miss")
            return result

        key = canonical_key(normalized)

        cached = self._cache.get(key)
        if cached is not None:
            self._stats["cache_hits"] += 1
            self._log(cached, key, 0, started, FetchTiming(), cache="hit")
            return dict(cached)  # copy: callers must not mutate the cache

        if self._respect_robots and not self.robots_allowed(normalized):
            self._stats["robots_blocked"] += 1
            result = self._failure(url, normalized, 0, "blocked by robots.txt")
            self._cache[key] = result
            self._log(result, key, 0, started, FetchTiming(), cache="miss")
            return dict(result)

        timing = FetchTiming()
        result, link_count = self._fetch_live(url, normalized, timing)
        self._cache[key] = result
        self._stats["fetches"] += 1
        self._timing["rate_limit_wait_s"] += timing.wait_s
        self._timing["retry_wait_s"] += timing.retry_wait_s
        self._timing["request_s"] += timing.request_s
        self._timing["pdf_text_s"] += timing.pdf_text_s
        self._timing["html_parse_s"] += timing.html_parse_s
        self._timing["link_extract_s"] += timing.link_extract_s
        if not result["ok"]:
            self._stats["errors"] += 1
        self._log(result, key, link_count, started, timing, cache="miss")
        return dict(result)

    def robots_allowed(self, url: str) -> bool:
        """Check robots.txt, fetched once per origin per run.

        Fails *open* when robots.txt cannot be fetched or is missing (a 404
        means allow-all) and *closed* on an explicit Disallow. Subdomains get
        their own robots.txt, hence the per-origin cache.
        """
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        entry = self._robots.get(origin)
        if entry is None:
            entry = self._load_robots(origin)
            self._robots[origin] = entry
        parser, _ = entry
        allowed = parser.can_fetch(config.USER_AGENT, url)
        if not allowed:
            logger.warning("robots.txt disallows %s -- not fetching", url)
        return allowed

    # -- internals ---------------------------------------------------------
    def _load_robots(self, origin: str) -> tuple[RobotFileParser, float]:
        parser = RobotFileParser()
        parser.set_url(f"{origin}/robots.txt")
        crawl_delay = 0.0
        robots_started = time.perf_counter()
        try:
            response = self._session.get(
                f"{origin}/robots.txt", timeout=min(self._timeout, 10.0), allow_redirects=True
            )
            if response.status_code >= 400:
                logger.info(
                    "robots.txt for %s returned HTTP %s -- treating as allow-all",
                    origin,
                    response.status_code,
                )
                parser.parse([])
            else:
                body = _decode(response.content, response.headers.get("Content-Type", ""))
                parser.parse(body.splitlines())
                logger.info("robots.txt loaded for %s", origin)
        except requests.RequestException as exc:
            logger.warning("robots.txt fetch failed for %s (%s) -- treating as allow-all", origin, exc)
            parser.parse([])

        self._timing["robots_s"] += time.perf_counter() - robots_started
        try:
            declared = parser.crawl_delay(config.USER_AGENT)
            crawl_delay = float(declared) if declared else 0.0
        except (AttributeError, TypeError, ValueError):
            crawl_delay = 0.0
        if crawl_delay:
            logger.info("robots.txt declares crawl-delay=%.1fs for %s", crawl_delay, origin)
        return parser, crawl_delay

    def _crawl_delay(self, url: str) -> float:
        parts = urlsplit(url)
        entry = self._robots.get(f"{parts.scheme}://{parts.netloc}")
        return entry[1] if entry else 0.0

    def _get_with_retry(
        self, url: str, timing: FetchTiming
    ) -> tuple[requests.Response | None, str | None]:
        host = urlsplit(url).hostname or ""
        attempts = max(1, self._max_retries + 1)
        last_error: str | None = None

        for attempt in range(attempts):
            timing.wait_s += self._rate_limiter.wait(host, self._crawl_delay(url))
            retryable = False
            started = time.perf_counter()
            try:
                response = self._session.get(
                    url, timeout=self._timeout, allow_redirects=True, stream=True
                )
            except requests.Timeout as exc:
                timing.request_s += time.perf_counter() - started
                last_error, retryable = f"timeout after {self._timeout}s: {exc}", True
            except requests.RequestException as exc:
                last_error, retryable = f"request failed: {exc}", True
            else:
                timing.request_s += time.perf_counter() - started
                status = getattr(response, "status_code", 0)
                # 403/404 are settled answers; retrying only spends page budget.
                if status in config.RETRY_STATUS and attempt < attempts - 1:
                    response.close()
                    last_error, retryable = f"HTTP {status}", True
                else:
                    return response, None

            if retryable and attempt < attempts - 1:
                backoff = config.RETRY_BACKOFF_S * (2**attempt)
                logger.warning("retrying %s in %.1fs after %s", url, backoff, last_error)
                timing.retry_wait_s += backoff
                time.sleep(backoff)

        return None, last_error

    @staticmethod
    def _read_capped(response: requests.Response) -> bytes:
        declared = response.headers.get("Content-Length")
        if declared and declared.isdigit() and int(declared) > config.MAX_CONTENT_BYTES:
            raise ValueError(f"response too large ({declared} bytes)")
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            total += len(chunk)
            if total > config.MAX_CONTENT_BYTES:
                raise ValueError(f"response exceeded {config.MAX_CONTENT_BYTES} bytes")
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _classify(content_type_header: str, content: bytes) -> ContentKind:
        """Content-Type first, magic bytes as the tiebreaker.

        Sitecore serves PDFs from ``/-/media/...ashx`` paths with no useful
        extension, so the URL cannot decide this. The ``%PDF-`` signature
        catches the case where the header is generic (octet-stream) too.
        """
        header = (content_type_header or "").lower()
        if content[:5] == b"%PDF-" or "application/pdf" in header:
            return "pdf"
        if "html" in header or "xhtml" in header:
            return "html"
        if not header and b"<html" in content[:2048].lower():
            return "html"
        return "other"

    def _fetch_live(
        self, requested: str, url: str, timing: FetchTiming
    ) -> tuple[PageDict, int]:
        response, error = self._get_with_retry(url, timing)
        if response is None:
            return self._failure(requested, url, 0, error or "unknown fetch error"), 0

        read_started = time.perf_counter()
        try:
            content = self._read_capped(response)
        except ValueError as exc:
            return self._failure(requested, url, response.status_code, str(exc)), 0
        except requests.RequestException as exc:
            return self._failure(requested, url, response.status_code, f"read failed: {exc}"), 0
        finally:
            timing.request_s += time.perf_counter() - read_started
            response.close()

        status = response.status_code
        final_url = normalize_url(str(response.url), url, language=self.language) or url
        header = response.headers.get("Content-Type", "")
        kind = self._classify(header, content)

        if status >= 400:
            return (
                self._failure(requested, final_url, status, f"HTTP {status}", content_type=kind),
                0,
            )

        if kind == "pdf":
            self._stats["pdfs"] += 1
            pdf_started = time.perf_counter()
            text, pdf_error = pdf_to_text(content)
            timing.pdf_text_s += time.perf_counter() - pdf_started
            return (
                PageDict(
                    url=final_url,
                    requested_url=requested,
                    status=status,
                    ok=text is not None,
                    content_type="pdf",
                    raw_html=None,
                    text=text,
                    fetched_at=_now(),
                    render_mode="requests",
                    error=pdf_error,
                ),
                0,
            )

        if kind != "html":
            return (
                PageDict(
                    url=final_url,
                    requested_url=requested,
                    status=status,
                    ok=False,
                    content_type="other",
                    raw_html=None,
                    text=None,
                    fetched_at=_now(),
                    render_mode="requests",
                    error=f"unsupported content type: {header or 'unknown'}",
                ),
                0,
            )

        raw_html = _decode(content, header)
        parse_started = time.perf_counter()
        text = html_to_text(raw_html)
        timing.html_parse_s += time.perf_counter() - parse_started
        link_started = time.perf_counter()
        links = extract_links(raw_html, final_url, language=self.language)
        timing.link_extract_s += time.perf_counter() - link_started
        raw_html, text, link_count, render_mode = self._maybe_escalate(
            final_url, raw_html, text, len(links)
        )

        return (
            PageDict(
                url=final_url,
                requested_url=requested,
                status=status,
                ok=True,
                content_type="html",
                raw_html=raw_html,
                text=text,
                fetched_at=_now(),
                render_mode=render_mode,
                error=None,
            ),
            link_count,
        )

    # -- render escalation -------------------------------------------------
    def _maybe_escalate(
        self, url: str, raw_html: str, text: str, link_count: int
    ) -> tuple[str, str, int, RenderMode]:
        reason = escalation_reason(raw_html, text, link_count)
        if reason is None:
            return raw_html, text, link_count, "requests"

        if not self._allow_playwright:
            logger.info("under-rendered page kept as-is (playwright disabled): %s [%s]", url, reason)
            return raw_html, text, link_count, "requests"

        logger.info("escalating to playwright: %s [reason=%s text=%d links=%d]", url, reason, len(text), link_count)
        self._stats["escalations"] += 1

        rendered, error = self._render_with_playwright(url)
        if rendered is None:
            logger.warning("playwright escalation failed for %s: %s", url, error)
            return raw_html, text, link_count, "requests"

        new_text = html_to_text(rendered)
        new_count = len(extract_links(rendered, url, language=self.language))
        if len(new_text) <= len(text) and new_count <= link_count:
            # A browser render that is no better means the page really is thin,
            # or we hit a bot wall. Keeping the requests result means
            # render_mode always names what was actually returned.
            logger.info(
                "playwright escalation did not help %s (text %d->%d, links %d->%d)",
                url, len(text), len(new_text), link_count, new_count,
            )
            return raw_html, text, link_count, "requests"

        self._stats["escalations_helped"] += 1
        logger.info(
            "playwright escalation improved %s (text %d->%d, links %d->%d)",
            url, len(text), len(new_text), link_count, new_count,
        )
        return rendered, new_text, new_count, "playwright"

    def _render_with_playwright(self, url: str) -> tuple[str | None, str | None]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            # Optional dependency: the project must still run for someone who
            # has not installed browsers.
            logger.warning(
                "playwright is not installed -- keeping the requests result for %s. "
                "Install with `pip install playwright && playwright install chromium`.",
                url,
            )
            self._allow_playwright = False  # do not retry the import per page
            return None, "playwright not installed"

        try:
            with sync_playwright() as driver:
                browser = driver.chromium.launch(headless=True)
                try:
                    context = browser.new_context(user_agent=config.USER_AGENT)
                    page = context.new_page()
                    page.goto(url, wait_until="domcontentloaded", timeout=int(self._timeout * 1000))
                    try:
                        page.wait_for_load_state(
                            "networkidle", timeout=config.PLAYWRIGHT_NETWORKIDLE_MS
                        )
                    except Exception:
                        # networkidle never settles on pages that poll in the
                        # background; the DOM is usually ready regardless.
                        pass
                    return page.content(), None
                finally:
                    browser.close()
        except Exception as exc:
            return None, f"playwright error: {exc}"

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _failure(
        requested: str,
        url: str,
        status: int,
        error: str,
        content_type: ContentKind = "other",
    ) -> PageDict:
        return PageDict(
            url=url,
            requested_url=requested,
            status=status,
            ok=False,
            content_type=content_type,
            raw_html=None,
            text=None,
            fetched_at=_now(),
            render_mode="requests",
            error=error,
        )

    @staticmethod
    def _log(
        result: PageDict, key: str, link_count: int, started: float,
        timing: FetchTiming, *, cache: str,
    ) -> None:
        """One structured line per fetch -- part of the deliverable audit trail.

        The canonical key is logged next to the URL so that a dedupe bug (two
        spellings of one page, or a token that survived stripping) is visible
        directly in the step log instead of needing a reproduction.

        ``wait_ms``/``retry_ms``/``req_ms``/``parse_ms``/``pdf_ms`` break the
        total down, so a slow hop can be attributed to our own pacing, to
        backing off after a failure, to the site itself, or to the work done on
        the bytes once they arrive -- without guessing.
        """
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "fetch url=%s key=%s status=%s ok=%s type=%s render=%s cache=%s links=%d text=%d "
            "wait_ms=%d retry_ms=%d req_ms=%d parse_ms=%d pdf_ms=%d ms=%d%s",
            result["url"],
            key,
            result["status"],
            result["ok"],
            result["content_type"],
            result["render_mode"],
            cache,
            link_count,
            len(result["text"] or ""),
            int(timing.wait_s * 1000),
            int(timing.retry_wait_s * 1000),
            int(timing.request_s * 1000),
            int((timing.html_parse_s + timing.link_extract_s) * 1000),
            int(timing.pdf_text_s * 1000),
            elapsed_ms,
            f" error={result['error']!r}" if result["error"] else "",
        )


def escalation_reason(raw_html: str, text: str, link_count: int) -> str | None:
    """Decide whether a requests-fetched page looks under-rendered.

    A survey of the site found ~1931 pages fine with plain requests against ~65
    needing a browser, so escalation has to be evidence-driven: rendering
    everything would pay browser cost on ~97% of fetches for nothing. Returns a
    reason code for the log, or None to keep the requests result.
    """
    soup = make_soup(raw_html)
    body = soup.body
    if body is None or not _clean_inline(body.get_text(" ", strip=True)):
        return "C:empty-body"
    if len(text) < config.ESCALATE_MIN_TEXT_CHARS:
        return "B:thin-text"
    if any(soup.find(id=marker) is not None for marker in config.SPA_SHELL_IDS) and len(
        text
    ) < config.ESCALATE_MIN_TEXT_CHARS * config.ESCALATE_SHELL_TEXT_MULTIPLIER:
        return "D:spa-shell"
    # Weakest signal, checked last: the nav renders into every server-rendered
    # page, so a page with almost no links genuinely is a shell.
    if link_count < config.ESCALATE_MIN_LINKS:
        return "A:few-links"
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Module-level convenience API
# --------------------------------------------------------------------------
def fetch_page(url: str, session: requests.Session | Fetcher | None = None) -> PageDict:
    """Fetch one page.

    Pass a :class:`Fetcher` to share its per-run cache, politeness clock and
    counters -- the agent loop should always do this. Passing a bare
    ``requests.Session`` (or nothing) creates a throwaway ``Fetcher``, which is
    fine for scripts and one-off calls but caches nothing across calls.
    """
    if isinstance(session, Fetcher):
        return session.fetch(url)
    return Fetcher(session=session).fetch(url)


def robots_allowed(url: str, session: requests.Session | Fetcher | None = None) -> bool:
    """Check robots.txt for a URL. See :meth:`Fetcher.robots_allowed`."""
    if isinstance(session, Fetcher):
        return session.robots_allowed(url)
    return Fetcher(session=session).robots_allowed(url)


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    # Reads saved fixtures, never the live site.
    #
    #   python -m browsing.fetcher                       # synthetic fixtures
    #   python -m browsing.fetcher fixtures/live         # a directory
    #   python -m browsing.fetcher fixtures/live/index.html
    #   python -m browsing.fetcher "fixtures/live/*.html"
    import pathlib
    import sys

    from browsing._demo import FixtureNotFound, base_url_for, resolve_fixture_paths

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    root = pathlib.Path(__file__).resolve().parent.parent
    try:
        files = resolve_fixture_paths(sys.argv[1:], root / "fixtures" / "synthetic")
    except FixtureNotFound as exc:
        print(exc)
        raise SystemExit(1) from None

    for path in files:
        raw = path.read_text(encoding="utf-8", errors="replace")
        base = base_url_for(path, "https://www.banquemisr.com/")
        text = html_to_text(raw)
        count = len(extract_links(raw, base))
        reason = escalation_reason(raw, text, count)
        print(f"\n=== {path.name} ===")
        print(f"  text={len(text)} chars  links={count}  escalate={reason or 'no'}")
        preview = "\n".join(f"  | {line}" for line in text.splitlines()[:12])
        print(preview)
