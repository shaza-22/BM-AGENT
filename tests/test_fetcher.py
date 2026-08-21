"""Tests for fetching, text rendering, PDF handling, caching and escalation.

No network: every response comes from ``FakeSession``. The politeness delay is
set to zero so the suite stays fast; the delay itself is tested separately on
:class:`RateLimiter`.
"""

from __future__ import annotations

import logging
import time

import requests

from browsing import config
from browsing.fetcher import (
    Fetcher,
    RateLimiter,
    escalation_reason,
    fetch_page,
    html_to_text,
    pdf_to_text,
)
from conftest import FakeResponse, FakeSession, html_response, minimal_pdf

HOME = "https://www.banquemisr.com/"
PAGE = "https://www.banquemisr.com/Home/Pages/Fees"


def make_fetcher(routes, **kwargs):
    kwargs.setdefault("respect_robots", False)
    kwargs.setdefault("delay_range", (0.0, 0.0))
    kwargs.setdefault("allow_playwright", False)
    session = FakeSession(routes)
    return Fetcher(session=session, **kwargs), session


def padded(body: str) -> str:
    """Wrap markup so it clears the under-rendered thresholds."""
    filler = "<p>" + ("Banque Misr information paragraph. " * 40) + "</p>"
    links = "".join(f'<a href="/Home/Pages/Filler{i}">Filler {i}</a>' for i in range(8))
    return f"<html><body>{body}{filler}{links}</body></html>"


# --------------------------------------------------------------------------
class TestHtmlToText:
    def test_table_rows_and_cells_keep_their_delimiters(self):
        html = """
        <table>
          <tr><th>Fee type</th><th>Amount</th></tr>
          <tr><td>Annual fee</td><td>300 EGP</td></tr>
          <tr><td>Cash withdrawal</td><td>2%</td></tr>
        </table>
        """
        text = html_to_text(html)
        lines = [line for line in text.splitlines() if line.strip()]
        assert "Fee type | Amount" in lines
        assert "Annual fee | 300 EGP" in lines
        assert "Cash withdrawal | 2%" in lines
        # The failure this guards against: every cell run together on one line,
        # so no reader can tell which amount belongs to which fee.
        assert "Annual fee300 EGP" not in text
        assert "300 EGPCash withdrawal" not in text

    def test_header_row_gets_a_separator(self):
        html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        lines = html_to_text(html).splitlines()
        assert lines[:3] == ["A | B", "--- | ---", "1 | 2"]

    def test_table_without_a_header_row_is_still_delimited(self):
        html = "<table><tr><td>Annual fee</td><td>300 EGP</td></tr></table>"
        assert "Annual fee | 300 EGP" in html_to_text(html)

    def test_each_row_is_on_its_own_line(self):
        html = (
            "<table><tr><td>Annual fee</td><td>300</td></tr>"
            "<tr><td>Late payment</td><td>150</td></tr></table>"
        )
        rows = [line for line in html_to_text(html).splitlines() if " | " in line]
        assert rows == ["Annual fee | 300", "Late payment | 150"]

    def test_nested_tables_are_rendered_inside_out(self):
        html = """
        <table><tr><td>Outer</td>
          <td><table><tr><td>Inner A</td><td>Inner B</td></tr></table></td>
        </tr></table>
        """
        text = html_to_text(html)
        assert "Inner A | Inner B" in text

    def test_empty_cells_hold_their_column_position(self):
        html = "<table><tr><td>Annual fee</td><td></td><td>Waived</td></tr></table>"
        row = html_to_text(html).splitlines()[0]
        # The middle column is empty but still countable, so a reader can tell
        # which column a value is missing from.
        assert [cell.strip() for cell in row.split("|")] == ["Annual fee", "", "Waived"]

    def test_scripts_styles_and_comments_are_removed(self):
        html = """
        <body>
          <script>var secret = "should not appear";</script>
          <style>.x { color: red; }</style>
          <noscript>Enable JavaScript</noscript>
          <!-- hidden comment -->
          <p>Visible text</p>
        </body>
        """
        text = html_to_text(html)
        assert "Visible text" in text
        for unwanted in ("secret", "color: red", "Enable JavaScript", "hidden comment"):
            assert unwanted not in text

    def test_whitespace_is_collapsed_without_merging_blocks(self):
        html = "<p>First    paragraph</p>\n\n<p>Second     paragraph</p>"
        assert html_to_text(html).splitlines() == ["First paragraph", "Second paragraph"]

    def test_line_breaks_separate_text(self):
        assert html_to_text("<p>Line one<br/>Line two</p>").splitlines() == ["Line one", "Line two"]

    def test_empty_input(self):
        assert html_to_text("") == ""

    def test_fixture_fee_table_survives(self, fixture_html):
        text = html_to_text(fixture_html("product_page.html"))
        assert "Annual fee | 300 EGP | Waived first year" in text
        assert "should not appear in text" not in text


# --------------------------------------------------------------------------
class TestEscalationHeuristic:
    def test_healthy_page_is_not_escalated(self):
        html = padded("<h1>Fees</h1>")
        assert escalation_reason(html, html_to_text(html), 40) is None

    def test_empty_body(self):
        assert escalation_reason("<html><body></body></html>", "", 0) == "C:empty-body"

    def test_missing_body(self):
        assert escalation_reason("<html></html>", "", 0) == "C:empty-body"

    def test_thin_text(self):
        html = "<html><body><p>Short.</p></body></html>"
        assert escalation_reason(html, html_to_text(html), 40) == "B:thin-text"

    def test_spa_shell_marker(self, fixture_html):
        html = fixture_html("spa_shell.html")
        assert escalation_reason(html, html_to_text(html), 0) in ("C:empty-body", "D:spa-shell")

    def test_shell_marker_with_real_content_is_left_alone(self):
        html = padded('<div id="app"><h1>Fees</h1></div>')
        assert escalation_reason(html, html_to_text(html), 40) is None

    def test_few_links_is_the_last_resort_signal(self):
        html = padded("<h1>Fees</h1>")
        assert escalation_reason(html, html_to_text(html), 2) == "A:few-links"

    def test_thresholds_come_from_config(self, monkeypatch):
        monkeypatch.setattr(config, "ESCALATE_MIN_TEXT_CHARS", 10_000)
        html = padded("<h1>Fees</h1>")
        assert escalation_reason(html, html_to_text(html), 40) == "B:thin-text"


class TestPlaywrightIsOptional:
    def test_missing_playwright_returns_the_requests_result(self, monkeypatch, caplog):
        # The project must still run for someone who has not installed browsers.
        monkeypatch.setitem(__import__("sys").modules, "playwright.sync_api", None)
        thin = "<html><body><p>Short page.</p></body></html>"
        fetcher, _ = make_fetcher({PAGE: html_response(PAGE, thin)}, allow_playwright=True)
        with caplog.at_level(logging.WARNING):
            result = fetcher.fetch(PAGE)
        assert result["ok"] is True
        assert result["render_mode"] == "requests"
        assert "playwright is not installed" in caplog.text
        assert fetcher.stats["escalations"] == 1
        assert fetcher.stats["escalations_helped"] == 0

    def test_escalation_is_skipped_when_disabled(self, caplog):
        thin = "<html><body><p>Short page.</p></body></html>"
        fetcher, _ = make_fetcher({PAGE: html_response(PAGE, thin)})
        with caplog.at_level(logging.INFO):
            fetcher.fetch(PAGE)
        assert fetcher.stats["escalations"] == 0
        assert "playwright disabled" in caplog.text


# --------------------------------------------------------------------------
class TestFetchHtml:
    def test_successful_fetch(self):
        html = padded("<h1>Fees and Commissions</h1>")
        fetcher, _ = make_fetcher({PAGE: html_response(PAGE, html)})
        result = fetcher.fetch(PAGE)
        assert result["ok"] is True
        assert result["status"] == 200
        assert result["content_type"] == "html"
        assert result["render_mode"] == "requests"
        assert result["error"] is None
        assert result["requested_url"] == PAGE
        assert "Fees and Commissions" in result["text"]
        # Raw HTML is kept verbatim for downstream table extraction.
        assert "<h1>" in result["raw_html"]
        assert result["fetched_at"].endswith("+00:00")

    def test_requested_url_is_normalised_before_fetching(self):
        messy = PAGE + "?csrt=abc123#tariff"
        fetcher, session = make_fetcher({PAGE: html_response(PAGE, padded("<h1>Fees</h1>"))})
        result = fetcher.fetch(messy)
        assert session.calls == [PAGE]
        assert result["url"] == PAGE
        assert result["requested_url"] == messy

    def test_redirects_report_the_final_url(self):
        final = "https://www.banquemisr.com/Home/Pages/Fees/Current"
        response = html_response(final, padded("<h1>Fees</h1>"))
        fetcher, _ = make_fetcher({PAGE: response})
        assert fetcher.fetch(PAGE)["url"] == final

    def test_utf8_is_decoded_without_a_charset_header(self):
        body = padded("<p>Rate – 5 % — Ma’moura branch</p>")
        response = FakeResponse(PAGE, content=body.encode("utf-8"), content_type="text/html")
        fetcher, _ = make_fetcher({PAGE: response})
        text = fetcher.fetch(PAGE)["text"]
        assert "–" in text and "—" in text
        assert "Ã" not in text  # the mojibake signature of an ISO-8859-1 fallback


class TestFailuresNeverRaise:
    def test_404_is_reported_not_raised(self):
        fetcher, session = make_fetcher({PAGE: FakeResponse(PAGE, status_code=404)})
        result = fetcher.fetch(PAGE)
        assert result["ok"] is False
        assert result["status"] == 404
        assert "404" in result["error"]
        assert len(session.calls) == 1  # a definitive answer is never retried

    def test_403_is_not_retried(self):
        fetcher, session = make_fetcher({PAGE: FakeResponse(PAGE, status_code=403)})
        assert fetcher.fetch(PAGE)["ok"] is False
        assert len(session.calls) == 1

    def test_503_is_retried_once_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _seconds: None)
        routes = {
            PAGE: [
                FakeResponse(PAGE, status_code=503),
                html_response(PAGE, padded("<h1>Fees</h1>")),
            ]
        }
        fetcher, session = make_fetcher(routes)
        result = fetcher.fetch(PAGE)
        assert result["ok"] is True
        assert len(session.calls) == 2

    def test_timeout_is_reported(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _seconds: None)
        fetcher, _ = make_fetcher({PAGE: requests.Timeout("timed out")})
        result = fetcher.fetch(PAGE)
        assert result["ok"] is False
        assert "timeout" in result["error"].lower()

    def test_connection_error_is_reported(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _seconds: None)
        fetcher, _ = make_fetcher({PAGE: requests.ConnectionError("refused")})
        assert fetcher.fetch(PAGE)["ok"] is False

    def test_off_domain_url_is_refused_without_a_request(self):
        fetcher, session = make_fetcher({})
        result = fetcher.fetch("https://evil-banquemisr.com/phish")
        assert result["ok"] is False
        assert "normalize_url" in result["error"]
        assert session.calls == []

    def test_oversized_response_is_refused(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_CONTENT_BYTES", 100)
        response = FakeResponse(PAGE, content=b"x" * 500, content_type="text/html")
        fetcher, _ = make_fetcher({PAGE: response})
        result = fetcher.fetch(PAGE)
        assert result["ok"] is False
        assert "exceeded" in result["error"]


class TestContentTypes:
    def test_pdf_by_content_type(self):
        pdf = minimal_pdf(["Tariff of charges", "Annual fee 300 EGP"])
        url = "https://www.banquemisr.com/Home/Pages/tariff.pdf"
        response = FakeResponse(url, content=pdf, content_type="application/pdf")
        fetcher, _ = make_fetcher({url: response})
        result = fetcher.fetch(url)
        assert result["ok"] is True
        assert result["content_type"] == "pdf"
        assert result["raw_html"] is None
        assert "Annual fee 300 EGP" in result["text"]
        assert fetcher.stats["pdfs"] == 1

    def test_pdf_by_magic_bytes_when_the_header_lies(self):
        # Sitecore serves PDFs from /-/media/....ashx, often as octet-stream.
        pdf = minimal_pdf(["Tariff of charges"])
        url = "https://www.banquemisr.com/-/media/Fees/tariff.ashx"
        response = FakeResponse(url, content=pdf, content_type="application/octet-stream")
        fetcher, _ = make_fetcher({url: response})
        result = fetcher.fetch(url)
        assert result["content_type"] == "pdf"
        assert "Tariff of charges" in result["text"]

    def test_broken_pdf_reports_an_error_instead_of_raising(self):
        url = "https://www.banquemisr.com/Home/Pages/broken.pdf"
        response = FakeResponse(url, content=b"%PDF-1.4\ntruncated", content_type="application/pdf")
        fetcher, _ = make_fetcher({url: response})
        result = fetcher.fetch(url)
        assert result["ok"] is False
        assert result["content_type"] == "pdf"
        assert result["error"]

    def test_unsupported_content_type(self):
        url = "https://www.banquemisr.com/Home/Pages/data.zip"
        response = FakeResponse(url, content=b"PK\x03\x04", content_type="application/zip")
        fetcher, _ = make_fetcher({url: response})
        result = fetcher.fetch(url)
        assert result["ok"] is False
        assert result["content_type"] == "other"

    def test_pdf_tables_are_delimited(self):
        pdf = minimal_pdf(["Annual fee 300 EGP"])
        text, error = pdf_to_text(pdf)
        assert error is None
        assert "Annual fee 300 EGP" in text


# --------------------------------------------------------------------------
class TestPerRunCache:
    def test_second_fetch_of_the_same_page_is_served_from_cache(self):
        fetcher, session = make_fetcher({PAGE: html_response(PAGE, padded("<h1>Fees</h1>"))})
        first = fetcher.fetch(PAGE)
        second = fetcher.fetch(PAGE)
        assert len(session.calls) == 1
        assert first["text"] == second["text"]
        assert fetcher.stats["cache_hits"] == 1

    def test_token_variants_hit_the_same_cache_entry(self):
        # The whole point of canonical keying: a fresh csrt token must not look
        # like a new page, or the page budget goes on re-fetches.
        fetcher, session = make_fetcher({PAGE: html_response(PAGE, padded("<h1>Fees</h1>"))})
        fetcher.fetch(PAGE + "?csrt=aaa")
        fetcher.fetch(PAGE + "?csrt=bbb")
        fetcher.fetch(PAGE.lower())
        assert len(session.calls) == 1
        assert fetcher.stats["cache_hits"] == 2

    def test_callers_cannot_mutate_the_cache(self):
        fetcher, _ = make_fetcher({PAGE: html_response(PAGE, padded("<h1>Fees</h1>"))})
        first = fetcher.fetch(PAGE)
        first["text"] = "tampered"
        assert fetcher.fetch(PAGE)["text"] != "tampered"

    def test_a_new_fetcher_starts_empty(self):
        # Nothing persists across tasks: every task re-reads the live site.
        routes = {PAGE: html_response(PAGE, padded("<h1>Fees</h1>"))}
        first, session_one = make_fetcher(routes)
        first.fetch(PAGE)
        second, session_two = make_fetcher(routes)
        second.fetch(PAGE)
        assert len(session_one.calls) == 1 and len(session_two.calls) == 1
        assert second.stats["cache_hits"] == 0

    def test_clear_cache(self):
        fetcher, session = make_fetcher({PAGE: html_response(PAGE, padded("<h1>Fees</h1>"))})
        fetcher.fetch(PAGE)
        fetcher.clear_cache()
        fetcher.fetch(PAGE)
        assert len(session.calls) == 2


class TestRobots:
    ROBOTS = "https://www.banquemisr.com/robots.txt"

    def _fetcher(self, robots_body: object):
        session = FakeSession({self.ROBOTS: robots_body, PAGE: html_response(PAGE, padded("<h1>F</h1>"))})
        return Fetcher(session=session, respect_robots=True, delay_range=(0.0, 0.0),
                       allow_playwright=False), session

    def test_disallowed_page_is_not_fetched(self):
        body = html_response(self.ROBOTS, "User-agent: *\nDisallow: /Home/Pages/", content_type="text/plain")
        fetcher, session = self._fetcher(body)
        result = fetcher.fetch(PAGE)
        assert result["ok"] is False
        assert "robots" in result["error"]
        assert PAGE not in session.calls
        assert fetcher.stats["robots_blocked"] == 1

    def test_allowed_page_is_fetched(self):
        body = html_response(self.ROBOTS, "User-agent: *\nDisallow: /private/", content_type="text/plain")
        fetcher, session = self._fetcher(body)
        assert fetcher.fetch(PAGE)["ok"] is True
        assert PAGE in session.calls

    def test_missing_robots_txt_fails_open(self):
        fetcher, _ = self._fetcher(FakeResponse(self.ROBOTS, status_code=404))
        assert fetcher.fetch(PAGE)["ok"] is True

    def test_unreachable_robots_txt_fails_open(self):
        fetcher, _ = self._fetcher(requests.ConnectionError("no route"))
        assert fetcher.fetch(PAGE)["ok"] is True

    def test_robots_txt_is_fetched_once_per_run(self):
        body = html_response(self.ROBOTS, "User-agent: *\nAllow: /", content_type="text/plain")
        fetcher, session = self._fetcher(body)
        fetcher.fetch(PAGE)
        fetcher.fetch("https://www.banquemisr.com/Home/Pages/Other")
        assert session.calls.count(self.ROBOTS) == 1


class TestRateLimiter:
    def test_delay_is_enforced_per_host(self):
        limiter = RateLimiter((0.05, 0.05))
        limiter.wait("www.banquemisr.com")
        started = time.monotonic()
        limiter.wait("www.banquemisr.com")
        assert time.monotonic() - started >= 0.04

    def test_different_hosts_do_not_block_each_other(self):
        limiter = RateLimiter((0.2, 0.2))
        limiter.wait("www.banquemisr.com")
        started = time.monotonic()
        limiter.wait("digital.banquemisr.com")
        assert time.monotonic() - started < 0.1

    def test_robots_crawl_delay_is_honoured_when_longer(self):
        limiter = RateLimiter((0.0, 0.0))
        limiter.wait("host")
        started = time.monotonic()
        limiter.wait("host", min_delay=0.05)
        assert time.monotonic() - started >= 0.04


class TestTimingVisibility:
    """A slow run must be attributable without a stopwatch.

    Politeness pacing, retry backoff and a slow server look identical from
    outside, and only one of them is ours to change.
    """

    def test_rate_limit_waits_are_logged(self, caplog):
        limiter = RateLimiter((0.05, 0.05))
        limiter.wait("www.banquemisr.com")
        with caplog.at_level(logging.INFO, logger="browsing.fetcher"):
            limiter.wait("www.banquemisr.com")
        assert "rate-limit wait" in caplog.text
        assert "not a retry" in caplog.text
        assert "host=www.banquemisr.com" in caplog.text

    def test_no_log_when_no_wait_was_needed(self, caplog):
        with caplog.at_level(logging.INFO, logger="browsing.fetcher"):
            RateLimiter((0.0, 0.0)).wait("host")
        assert "rate-limit wait" not in caplog.text

    def test_the_fetch_line_separates_waiting_from_the_request(self, caplog):
        # The old line reported one total, so a 7s hop could have been 6s of
        # server time or 6s of our own pacing and there was no way to tell.
        fetcher, _ = make_fetcher({PAGE: html_response(PAGE, padded("<h1>Fees</h1>"))},
                                  delay_range=(0.05, 0.05))
        fetcher.fetch(PAGE)
        with caplog.at_level(logging.INFO, logger="browsing.fetcher"):
            fetcher.fetch("https://www.banquemisr.com/Home/Pages/Other")
        assert "wait_ms=" in caplog.text
        assert "req_ms=" in caplog.text
        assert "retry_ms=" in caplog.text

    def test_stats_report_where_the_time_went(self):
        fetcher, _ = make_fetcher({PAGE: html_response(PAGE, padded("<h1>Fees</h1>"))},
                                  delay_range=(0.05, 0.05))
        fetcher.fetch(PAGE)
        fetcher.fetch("https://www.banquemisr.com/Home/Pages/Other")
        stats = fetcher.stats
        assert stats["rate_limit_wait_s"] >= 0.04     # the second fetch waited
        assert stats["retry_wait_s"] == 0.0
        assert "request_s" in stats and "robots_s" in stats

    def test_retry_backoff_is_counted_separately_from_pacing(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _seconds: None)
        routes = {PAGE: [FakeResponse(PAGE, status_code=503),
                         html_response(PAGE, padded("<h1>Fees</h1>"))]}
        fetcher, _ = make_fetcher(routes)
        fetcher.fetch(PAGE)
        assert fetcher.stats["retry_wait_s"] == config.RETRY_BACKOFF_S
        assert fetcher.stats["rate_limit_wait_s"] == 0.0

    def test_robots_time_is_attributed(self):
        # A WAF that stalls robots.txt would otherwise be invisible.
        robots = "https://www.banquemisr.com/robots.txt"
        session = FakeSession({robots: html_response(robots, "User-agent: *\nAllow: /",
                                                     content_type="text/plain"),
                               PAGE: html_response(PAGE, padded("<h1>F</h1>"))})
        fetcher = Fetcher(session=session, respect_robots=True, delay_range=(0.0, 0.0),
                          allow_playwright=False)
        fetcher.fetch(PAGE)
        assert fetcher.stats["robots_s"] >= 0.0


class TestModuleApi:
    def test_fetch_page_shares_a_fetchers_cache(self):
        fetcher, session = make_fetcher({PAGE: html_response(PAGE, padded("<h1>Fees</h1>"))})
        fetch_page(PAGE, session=fetcher)
        fetch_page(PAGE, session=fetcher)
        assert len(session.calls) == 1

    def test_fetch_page_accepts_a_bare_session(self):
        session = FakeSession({PAGE: html_response(PAGE, padded("<h1>Fees</h1>"))})
        assert fetch_page(PAGE, session=session)["ok"] is True

    def test_log_line_carries_the_canonical_key(self, caplog):
        fetcher, _ = make_fetcher({PAGE: html_response(PAGE, padded("<h1>Fees</h1>"))})
        with caplog.at_level(logging.INFO, logger="browsing.fetcher"):
            fetcher.fetch(PAGE)
        assert "key=banquemisr.com/home/pages/fees" in caplog.text
        assert f"url={PAGE}" in caplog.text
        assert "render=requests" in caplog.text
