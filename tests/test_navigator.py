"""Tests for the navigation loop.

Two kinds of scenario: small hand-built sites, where the assertion can be exact,
and the real saved pages in ``fixtures/live/``, which carry the site's actual
scale (99 links on the homepage, a nav repeated on every page). Both run
offline -- pages come from a FakeSession, decisions from a FakeLLMClient.
"""

from __future__ import annotations

import io
import json
import pathlib
import logging

import pytest

from agent import config as agent_config
from agent.llm import FakeLLMClient, LLMError
from agent.navigator import Navigator, always_unresolved, navigate
from agent.trail_log import StepLogger
from browsing.fetcher import Fetcher
from conftest import FakeResponse, FakeSession, choose_by, live_fetcher, live_manifest

HOME = "https://www.banquemisr.com/"
CARDS = "https://www.banquemisr.com/Home/SMEs/Retail%20Banking/Pages/Cards"
LIST = "https://www.banquemisr.com/Home/SMEs/Retail%20Banking/Pages/Cards/Credit%20Cards%20List"

pytestmark = pytest.mark.filterwarnings("ignore")


LIVE_DIR = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "live"
LIVE_PDFS = {path.name: path for path in LIVE_DIR.glob("*.pdf")}


def has_live_fixtures() -> bool:
    return bool(live_manifest().get("pages"))


def readable_pdf() -> tuple[str, bytes]:
    """A saved PDF that actually yields text, with a URL to serve it at.

    Found by extraction rather than by name: the manifest's only PDF is the
    image-only ATM guide, and which document happens to be readable is a
    property of the snapshot, not something to hardcode.
    """
    from browsing.fetcher import pdf_to_text

    for name, path in sorted(LIVE_PDFS.items()):
        content = path.read_bytes()
        text, _ = pdf_to_text(content)
        if text and len(text) > 1000:
            return f"https://www.banquemisr.com/-/media/{name}", content
    pytest.skip("no saved PDF with extractable text")


needs_live = pytest.mark.skipif(not has_live_fixtures(), reason="fixtures/live/ is empty")


def page(*links: str, body: str = "Some page content about banking services.") -> str:
    anchors = "".join(f'<a href="{href}">{label}</a>' for href, label in (l.split("|") for l in links))
    return f"<html><body><main><p>{body}</p>{anchors}</main></body></html>"


def fetcher_for(pages: dict[str, str], **kwargs) -> Fetcher:
    routes = {url: FakeResponse(url, content=html.encode()) for url, html in pages.items()}
    kwargs.setdefault("respect_robots", False)
    kwargs.setdefault("delay_range", (0.0, 0.0))
    kwargs.setdefault("allow_playwright", False)
    return Fetcher(session=FakeSession(routes), **kwargs)


def reply(choice, reasoning="because", confidence=0.8, outcome=None) -> str:
    body = {"choice": choice, "reasoning": reasoning, "confidence": confidence}
    if outcome is not None:
        body["outcome"] = outcome
    return json.dumps(body)


def resolve_when(needle: str):
    def validate(sub_goal, page_dict):
        hit = needle.lower() in (page_dict.get("text") or "").lower()
        return {
            "resolved": hit,
            "extracted": {"url": page_dict["url"]} if hit else {},
            "reason": "found it" if hit else "not here",
        }

    return validate


# --------------------------------------------------------------------------
class TestResolution:
    def test_resolves_on_a_small_site(self):
        pages = {
            HOME: page("/answer|The Answer", "/other|Something Else"),
            "https://www.banquemisr.com/answer": page(body="The annual fee is 300 EGP."),
        }
        navigator = Navigator(
            FakeLLMClient([reply(0, "The Answer looks like the destination")]),
            fetcher=fetcher_for(pages),
            validate_fn=resolve_when("annual fee"),
        )
        result = navigator.navigate("find the annual fee")

        assert result.status == "resolved"
        assert result.page["url"].endswith("/answer")
        assert result.extracted == {"url": "https://www.banquemisr.com/answer"}
        assert result.hops_used == 1
        assert result.pages_fetched == 2

    def test_seed_is_always_the_first_step(self):
        navigator = Navigator(FakeLLMClient([reply(-1)]), fetcher=fetcher_for({HOME: page("/a|A")}))
        trail = navigator.navigate("anything").trail
        assert trail[0].url == HOME
        assert trail[0].hop == 0
        assert trail[0].label is None
        assert trail[0].reasoning  # the seed step still explains itself

    def test_sources_lists_only_pages_actually_fetched(self):
        pages = {HOME: page("/answer|Answer"), "https://www.banquemisr.com/answer": page(body="found")}
        result = Navigator(
            FakeLLMClient([reply(0)]),
            fetcher=fetcher_for(pages),
            validate_fn=resolve_when("found"),
        ).navigate("goal")
        assert result.sources == [HOME, "https://www.banquemisr.com/answer"]

    @needs_live
    def test_resolves_across_three_real_pages(self):
        navigator = Navigator(
            FakeLLMClient(choose_by("/Pages/Cards", "Credit%20Cards%20List")),
            fetcher=live_fetcher(),
            validate_fn=resolve_when("classic credit card"),
        )
        result = navigator.navigate("find the classic credit card")

        assert result.status == "resolved"
        assert result.hops_used == 2
        assert [step.links_found for step in result.trail][0] == 99  # the real homepage
        assert result.page["url"] == LIST


class TestCaps:
    def test_hop_cap_reports_which_cap_fired(self):
        pages = {HOME: page("/a|A", "/b|B"), "https://www.banquemisr.com/a": page("/b|B"),
                 "https://www.banquemisr.com/b": page("/a|A")}
        result = Navigator(
            FakeLLMClient(lambda prompt: reply(0)),
            fetcher=fetcher_for(pages), max_hops=2,
        ).navigate("never resolves")

        assert result.status == "exhausted"
        assert result.cap_hit == "hops"
        assert result.hops_used == 2
        assert "maximum of 2 hops" in result.final_reasoning

    def test_page_cap_reports_which_cap_fired(self):
        pages = {HOME: page("/a|A", "/b|B", "/c|C")}
        for name in "abc":
            pages[f"https://www.banquemisr.com/{name}"] = page("/a|A", "/b|B", "/c|C")
        result = Navigator(
            FakeLLMClient(lambda prompt: reply(0)),
            fetcher=fetcher_for(pages), max_hops=99, max_pages=3,
        ).navigate("never resolves")

        assert result.status == "exhausted"
        assert result.cap_hit == "pages"
        assert result.pages_fetched == 3

    def test_never_fabricates_a_page_on_exhaustion(self):
        result = Navigator(
            FakeLLMClient(lambda prompt: reply(0)),
            fetcher=fetcher_for({HOME: page("/a|A"), "https://www.banquemisr.com/a": page("/b|B"),
                                 "https://www.banquemisr.com/b": page()}),
            max_hops=1,
        ).navigate("goal")
        assert result.page is None
        assert result.extracted is None


class TestNoCandidates:
    def test_model_declining_ends_the_run_cleanly(self):
        llm = FakeLLMClient([reply(-1, "This site does not publish share prices.")])
        result = Navigator(llm, fetcher=fetcher_for({HOME: page("/a|A")})).navigate(
            "what is the share price of an unrelated company"
        )
        assert result.status == "no_candidates"
        assert "does not publish" in result.final_reasoning
        assert result.page is None

    def test_running_out_of_links_ends_the_run(self):
        result = Navigator(
            FakeLLMClient([reply(0), reply(0)]),
            fetcher=fetcher_for({HOME: page("/a|A"), "https://www.banquemisr.com/a": page()}),
        ).navigate("goal")
        assert result.status == "no_candidates"


class TestArrival:
    """Reaching the destination must not read as a failed run.

    Before this, "none of these links" covered both "nothing here is relevant"
    and "we have arrived, stop walking", so a successful walk reported
    no_candidates.
    """

    def test_arrival_is_its_own_status_and_returns_the_page(self):
        pages = {HOME: page("/a|Answer", "/b|Other"),
                 "https://www.banquemisr.com/a": page("/c|Elsewhere", body="the fee schedule")}
        llm = FakeLLMClient([
            reply(0, "Answer should be the destination"),
            reply(-1, "We are on the destination page for this sub-goal", outcome="arrived"),
        ])
        result = Navigator(llm, fetcher=fetcher_for(pages)).navigate("find the fees")

        assert result.status == "arrived"
        assert result.page["url"].endswith("/a")     # available for extraction
        assert "destination page" in result.final_reasoning

    def test_arrival_never_claims_the_sub_goal_is_answered(self):
        # One-way precedence: only validate_fn produces "resolved".
        pages = {HOME: page("/a|Answer", "/b|Other"),
                 "https://www.banquemisr.com/a": page("/c|Elsewhere", body="content")}
        llm = FakeLLMClient([reply(0), reply(-1, "we are here", outcome="arrived")])
        result = Navigator(llm, fetcher=fetcher_for(pages)).navigate("goal")

        assert result.status == "arrived"
        assert result.status != "resolved"
        assert result.extracted is None

    def test_a_validator_saying_resolved_still_wins(self):
        pages = {HOME: page("/a|Answer"), "https://www.banquemisr.com/a": page(body="the answer")}
        llm = FakeLLMClient([reply(0), reply(-1, "we are here", outcome="arrived")])
        result = Navigator(
            llm, fetcher=fetcher_for(pages), validate_fn=resolve_when("the answer")
        ).navigate("goal")

        assert result.status == "resolved"
        assert result.extracted == {"url": "https://www.banquemisr.com/a"}

    def test_nothing_relevant_is_still_no_candidates(self):
        llm = FakeLLMClient([reply(-1, "this site does not cover it", outcome="none")])
        result = Navigator(llm, fetcher=fetcher_for({HOME: page("/a|A")})).navigate("goal")
        assert result.status == "no_candidates"
        assert result.page is None

    def test_an_omitted_outcome_still_means_no_candidates(self):
        # Absent or unparseable defaults to "none", so an unexpected reply
        # degrades to the previous behaviour, never to a false arrival.
        llm = FakeLLMClient([reply(-1, "nothing fits")])
        result = Navigator(llm, fetcher=fetcher_for({HOME: page("/a|A")})).navigate("goal")
        assert result.status == "no_candidates"

    def test_arrival_needs_a_page_that_actually_loaded(self):
        # The seed itself 404s, so there is nothing to have arrived at.
        llm = FakeLLMClient([reply(-1, "we are here", outcome="arrived")])
        result = Navigator(llm, fetcher=fetcher_for({})).navigate("goal")
        assert result.status == "no_candidates"
        assert result.page is None

    def test_arrival_is_recorded_in_the_step_log(self):
        # An "arrived" run where the validator disagreed is the highest-value
        # row in the evaluation pipeline, so the outcome is logged per hop.
        pages = {HOME: page("/a|Answer", "/b|Other"),
                 "https://www.banquemisr.com/a": page("/c|Elsewhere", body="content")}
        navigator = Navigator(
            FakeLLMClient([reply(0), reply(-1, "here", outcome="arrived")]),
            fetcher=fetcher_for(pages), step_logger=StepLogger(run_id="r"),
        )
        navigator.navigate("goal")
        outcomes = [r["outcome"] for r in navigator.step_logger.records if r["event"] == "selection"]
        assert outcomes == ["follow", "arrived"]
        finished = navigator.step_logger.records[-1]
        assert finished["status"] == "arrived"

    def test_a_leaf_page_with_no_links_left_is_not_assumed_to_be_an_arrival(self):
        # With nothing to offer, the selector is never consulted, so there is
        # no arrival signal to act on. Reporting "arrived" here would be a
        # guess; the run says plainly that it ran out of links instead.
        pages = {HOME: page("/a|Answer"), "https://www.banquemisr.com/a": page(body="a leaf")}
        llm = FakeLLMClient([reply(0)])
        result = Navigator(llm, fetcher=fetcher_for(pages)).navigate("goal")

        assert result.status == "no_candidates"
        assert "No unvisited links remain" in result.final_reasoning

    @needs_live
    def test_arrival_on_a_real_page(self):
        navigator = Navigator(
            FakeLLMClient(choose_by("/Pages/Cards", "__stop__")),
            fetcher=live_fetcher(),
        )
        result = navigator.navigate("find the cards section")
        # choose_by returns choice -1 with no outcome when nothing matches, so
        # this is the plain no-candidate path on real pages.
        assert result.status == "no_candidates"
        assert result.hops_used == 1


class TestBacktracking:
    def test_a_dead_end_does_not_end_the_run(self):
        # /broken 404s; the run must recover using links from an earlier page.
        pages = {
            HOME: page("/broken|Broken", "/good|Good"),
            "https://www.banquemisr.com/good": page(body="the answer is here"),
        }
        llm = FakeLLMClient(choose_by("/broken", "/good"))
        result = Navigator(
            llm, fetcher=fetcher_for(pages), validate_fn=resolve_when("answer is here")
        ).navigate("goal")

        assert result.status == "resolved"
        assert result.pages_fetched == 3          # seed, dead end, answer
        assert [step.fetch_ok for step in result.trail] == [True, False, True]
        assert result.trail[1].fetch_status == 404

    def test_a_later_hop_can_follow_a_link_found_on_an_earlier_page(self):
        # The frontier's whole point: the answer link was on the homepage, but
        # the agent only realises it after the first branch turns out to be a
        # dead end in content terms. A strict tree-walk would have to spend a
        # hop backing up first.
        pages = {
            HOME: page("/detour|Detour", "/answer|Answer"),
            "https://www.banquemisr.com/detour": page(body="nothing relevant at all"),
            "https://www.banquemisr.com/answer": page(body="the annual fee is 300 EGP"),
        }
        llm = FakeLLMClient(choose_by("/detour", "/answer"))
        result = Navigator(
            llm, fetcher=fetcher_for(pages), validate_fn=resolve_when("annual fee")
        ).navigate("goal")

        assert result.status == "resolved"
        final = result.trail[-1]
        assert final.hop == 2
        assert final.discovered_at_hop == 0        # discovered on the seed page
        assert final.from_url == HOME

    def test_offered_but_unchosen_links_stay_selectable(self):
        pages = {
            HOME: page("/first|First", "/second|Second"),
            "https://www.banquemisr.com/first": page(body="wrong page"),
            "https://www.banquemisr.com/second": page(body="right page"),
        }
        llm = FakeLLMClient(choose_by("/first", "/second"))
        result = Navigator(
            llm, fetcher=fetcher_for(pages), validate_fn=resolve_when("right page")
        ).navigate("goal")
        assert result.status == "resolved"
        assert result.page["url"].endswith("/second")


class TestWafDetection:
    def test_block_page_aborts_the_run(self, fixture_html, caplog):
        routes = {HOME: FakeResponse(HOME, content=fixture_html("waf_block.html").encode())}
        fetcher = Fetcher(session=FakeSession(routes), respect_robots=False,
                          delay_range=(0.0, 0.0), allow_playwright=False)
        with caplog.at_level(logging.WARNING):
            result = Navigator(FakeLLMClient([reply(0)]), fetcher=fetcher).navigate("goal")

        assert result.status == "blocked"
        assert result.page is None
        assert "WAF block page" in caplog.text
        assert "ban" in result.final_reasoning

    def test_a_real_page_mentioning_the_words_is_not_mistaken_for_a_block(self):
        # Marker text alone is not enough: a real page carries 70+ links.
        body = "Access denied errors on your card can happen at some ATMs."
        links = [f"/p{i}|Page {i}" for i in range(10)]
        result = Navigator(
            FakeLLMClient([reply(-1)]), fetcher=fetcher_for({HOME: page(*links, body=body)})
        ).navigate("goal")
        assert result.status == "no_candidates"


class TestValidation:
    def test_default_stub_never_resolves(self):
        verdict = always_unresolved("goal", {"url": HOME, "text": "anything"})
        assert verdict["resolved"] is False
        assert "no validator installed" in verdict["reason"]

    def test_validator_receives_the_whole_page_dict(self):
        seen = {}

        def validate(sub_goal, page_dict):
            seen.update(page_dict)
            return {"resolved": True, "extracted": {}, "reason": "ok"}

        Navigator(FakeLLMClient([reply(0)]), fetcher=fetcher_for({HOME: page("/a|A")}),
                  validate_fn=validate).navigate("goal")

        # raw_html for tables, text for prose, url for attribution.
        assert set(seen) >= {"url", "text", "raw_html", "content_type", "status", "ok"}
        assert seen["raw_html"]

    def test_a_validator_that_raises_costs_one_page_not_the_task(self):
        def exploding(sub_goal, page_dict):
            raise ValueError("teammate bug")

        result = Navigator(
            FakeLLMClient([reply(-1)]), fetcher=fetcher_for({HOME: page("/a|A")}),
            validate_fn=exploding,
        ).navigate("goal")
        assert result.status == "no_candidates"       # ran on, did not crash
        assert "teammate bug" in result.trail[0].validate_reason

    def test_a_validator_returning_the_wrong_type_is_survivable(self):
        result = Navigator(
            FakeLLMClient([reply(-1)]), fetcher=fetcher_for({HOME: page("/a|A")}),
            validate_fn=lambda goal, page_dict: "yes",
        ).navigate("goal")
        assert result.status == "no_candidates"
        assert "expected dict" in result.trail[0].validate_reason

    def test_a_failed_fetch_is_never_reported_as_resolved(self):
        result = Navigator(
            FakeLLMClient([reply(-1)]),
            fetcher=fetcher_for({}),                   # seed itself 404s
            validate_fn=lambda goal, page_dict: {"resolved": True, "extracted": {}, "reason": "x"},
        ).navigate("goal")
        assert result.status != "resolved"

    @needs_live
    def test_a_pdf_can_resolve_a_sub_goal(self):
        # Fee data lives in PDFs linked from category pages, so a PDF has to be
        # a legitimate destination, not merely a link type. Uses the real
        # 23k-character fee schedule from fixtures/live/.
        url, content = readable_pdf()
        routes = {url: FakeResponse(url, content=content, content_type="application/pdf")}
        fetcher = Fetcher(session=FakeSession(routes), respect_robots=False,
                          delay_range=(0.0, 0.0), allow_playwright=False)
        result = Navigator(
            FakeLLMClient([reply(-1)]), fetcher=fetcher, seed_url=url,
            validate_fn=resolve_when("fee"),
        ).navigate("find the fee schedule")

        assert result.status == "resolved"
        assert result.page["content_type"] == "pdf"
        assert result.page["raw_html"] is None
        assert len(result.page["text"]) > 1000

    @needs_live
    def test_an_image_only_pdf_never_resolves(self):
        # The saved ATM guide is a scanned document: it fetches fine but yields
        # no text, so it must be treated as a dead end rather than an answer.
        path = LIVE_PDFS["media-guide-to-activate-debit-and-pre-paid-cards-through-the-atm-ashx.pdf"]
        url = "https://www.banquemisr.com/-/media/atm-guide.ashx"
        routes = {url: FakeResponse(url, content=path.read_bytes(), content_type="application/pdf")}
        fetcher = Fetcher(session=FakeSession(routes), respect_robots=False,
                          delay_range=(0.0, 0.0), allow_playwright=False)
        result = Navigator(
            FakeLLMClient([reply(-1)]), fetcher=fetcher, seed_url=url,
            validate_fn=lambda goal, p: {"resolved": True, "extracted": {}, "reason": "would resolve"},
        ).navigate("how do I activate a card at an ATM")

        assert result.status != "resolved"
        assert "no text" in result.trail[0].validate_reason.lower()


class TestAliasHandling:
    def test_following_an_alias_of_a_visited_page_is_warned_about(self, caplog):
        pages = {
            HOME: page("/en/ABOUT-US/History|History EN"),
            "https://www.banquemisr.com/en/ABOUT-US/History": page(
                "/Home/ABOUT%20US/History|History Home"
            ),
            "https://www.banquemisr.com/Home/ABOUT%20US/History": page(body="same content"),
        }
        llm = FakeLLMClient(choose_by("/en/ABOUT-US/History", "/Home/ABOUT%20US/History"))
        with caplog.at_level(logging.WARNING):
            result = Navigator(llm, fetcher=fetcher_for(pages)).navigate("history")

        assert "alternate URL scheme" in caplog.text
        # Deprioritised, not dropped -- the equivalence is only a heuristic.
        assert result.trail[-1].url.endswith("/Home/ABOUT%20US/History")


class TestTrailAndLogging:
    def test_every_step_carries_non_empty_reasoning(self):
        pages = {HOME: page("/a|A"), "https://www.banquemisr.com/a": page("/b|B"),
                 "https://www.banquemisr.com/b": page()}
        result = Navigator(
            FakeLLMClient(lambda prompt: reply(0)), fetcher=fetcher_for(pages)
        ).navigate("goal")
        assert result.trail
        for step in result.trail:
            assert step.reasoning.strip()

    def test_steps_are_emitted_as_json_lines(self):
        stream = io.StringIO()
        navigator = Navigator(
            FakeLLMClient([reply(0), reply(-1)]),
            fetcher=fetcher_for({HOME: page("/a|A"), "https://www.banquemisr.com/a": page()}),
            step_logger=StepLogger(stream, run_id="test-run"),
        )
        navigator.navigate("goal")

        records = [json.loads(line) for line in stream.getvalue().splitlines()]
        events = [record["event"] for record in records]
        assert events[0] == "navigation_started"
        assert events[-1] == "navigation_finished"
        assert "selection" in events and "step" in events
        assert all(record["run_id"] == "test-run" for record in records)

    def test_selection_records_carry_what_the_demo_needs(self):
        stream = io.StringIO()
        navigator = Navigator(
            FakeLLMClient([reply(0, "Cards should list the products")]),
            fetcher=fetcher_for({HOME: page("/a|Cards"), "https://www.banquemisr.com/a": page()}),
            step_logger=StepLogger(stream, run_id="r"),
        )
        navigator.navigate("goal")
        selection = next(
            json.loads(line) for line in stream.getvalue().splitlines()
            if json.loads(line)["event"] == "selection"
        )
        assert selection["reasoning"] == "Cards should list the products"
        assert selection["confidence"] == 0.8
        assert selection["chosen_label"] == "Cards"
        assert selection["candidates_offered"] >= 1
        assert selection["candidates_available"] >= 1

    @needs_live
    def test_candidate_trimming_is_reported_on_the_real_homepage(self):
        navigator = Navigator(FakeLLMClient([reply(-1)]), fetcher=live_fetcher())
        result = navigator.navigate("goal")
        selection = next(r for r in navigator.step_logger.records if r["event"] == "selection")
        assert selection["candidates_available"] > selection["candidates_offered"]
        assert selection["candidates_offered"] == agent_config.CANDIDATE_LIMIT
        assert result.trail[0].links_found == 99


class TestRunIsolation:
    def test_one_fetcher_serves_the_whole_run(self):
        pages = {HOME: page("/a|A"), "https://www.banquemisr.com/a": page("/b|B", "/a|A"),
                 "https://www.banquemisr.com/b": page()}
        fetcher = fetcher_for(pages)
        Navigator(FakeLLMClient(lambda prompt: reply(0)), fetcher=fetcher).navigate("goal")
        # Visited links are dropped before selection, so nothing is re-fetched.
        assert fetcher.stats["fetches"] == 3

    def test_exclude_urls_are_treated_as_already_visited(self):
        pages = {HOME: page("/skip|Skip", "/take|Take"),
                 "https://www.banquemisr.com/take": page(body="the answer")}
        llm = FakeLLMClient(choose_by("/take"))
        result = Navigator(
            llm, fetcher=fetcher_for(pages), validate_fn=resolve_when("the answer")
        ).navigate("goal", exclude_urls=["https://www.banquemisr.com/skip"])

        assert result.status == "resolved"
        prompt = llm.prompts[0]
        assert "/skip" not in prompt
        assert "/take" in prompt

    def test_llm_failure_ends_the_run_as_an_error(self):
        def explode(prompt):
            raise LLMError("no API key configured")

        result = Navigator(FakeLLMClient(explode), fetcher=fetcher_for({HOME: page("/a|A")})).navigate("goal")
        assert result.status == "error"
        assert "no API key" in result.final_reasoning

    def test_module_level_navigate_helper(self):
        result = navigate(
            "goal",
            llm=FakeLLMClient([reply(-1, "nothing fits")]),
            fetcher=fetcher_for({HOME: page("/a|A")}),
        )
        assert result.status == "no_candidates"
        assert result.sub_goal == "goal"
