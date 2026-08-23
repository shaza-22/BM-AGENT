"""The extraction fallback: reading a page when keyword matching found nothing.

The vendored extraction is keyword matching, and every visible failure of this
system falls out of that: Arabic pages yield nothing because there are no
Arabic keywords, a fees page reads as not-found because it words things
differently, a sub-goal resolves on product names and reports no verified
facts. Reading the page with a model is one general fix instead of four
special cases.

What keeps it safe is the verbatim rule: the label may be written, the value
may not. Most of this file is about that rule.
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from conftest import EXTRACTION_MARKER, PLANNING_MARKER, choose_by, copy_from_page, live_fetcher  # noqa: E402

from agent.extraction import Fact, extract_facts, normalise, verbatim_filter  # noqa: E402
from agent.llm import FakeLLMClient, LLMError  # noqa: E402
from agent.loop import ResearchLoop  # noqa: E402

FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "live"
PAGE = ("Fees and charges\nIssuance EGP 250\nRenewal EGP 250\n"
        "Supplementary cards issuance and renewal EGP100\nInterest rate 4% monthly\n")


def reply(facts, present=True, note=""):
    return json.dumps({"facts": facts, "present": present, "note": note})


class TestTheVerbatimRule:
    """The value must be on the page character for character."""

    def test_a_copied_value_is_kept(self):
        kept, struck = verbatim_filter([Fact("Issuance fee", "EGP 250")], PAGE)
        assert [f.value for f in kept] == ["EGP 250"] and struck == []

    @pytest.mark.parametrize("value,why", [
        ("250 EGP", "reordered"),
        ("EGP250", "space removed"),
        ("EGP 25", "a fragment of the real figure"),
        ("EGP 250.00", "reformatted"),
        ("EGP 500", "invented"),
        ("about EGP 250", "padded with a hedge"),
    ])
    def test_anything_not_copied_is_struck(self, value: str, why: str):
        """Discarded, never repaired -- repairing is deciding what the page meant."""
        kept, struck = verbatim_filter([Fact("Fee", value)], PAGE)
        assert kept == [] and len(struck) == 1, why

    def test_a_fragment_of_a_larger_number_does_not_pass(self):
        """A plain substring test passes "250" against a page saying "EGP 2500".

        That is a figure wrong by a factor of ten, carrying a citation. The
        match has to end where a token ends.
        """
        kept, struck = verbatim_filter([Fact("Fee", "250")], "Issuance EGP 2500 only")
        assert kept == [] and struck == [("Fee", "250")]

    def test_a_value_that_is_a_whole_token_still_matches(self):
        kept, _ = verbatim_filter(
            [Fact("a", "EGP 2500"), Fact("b", "4% monthly"), Fact("c", "(free)")],
            "Issuance EGP 2500\nInterest rate 4% monthly\nEnquiry (free)")
        assert len(kept) == 3

    def test_typography_is_folded_but_content_is_not(self):
        """A non-breaking space or an en dash is the same value; a digit is not."""
        page = "Issuance EGP 250 and a range 10–20"
        kept, _ = verbatim_filter([Fact("a", "EGP 250"), Fact("b", "10-20")], page)
        assert len(kept) == 2
        kept, struck = verbatim_filter([Fact("c", "EGP 251")], page)
        assert kept == [] and struck

    def test_arabic_values_match(self):
        page = "رسوم الإصدار 250 جنيه مصري\nرسوم التجديد 250 جنيه"
        kept, struck = verbatim_filter(
            [Fact("Issuance", "250 جنيه مصري"), Fact("Invented", "999 جنيه")], page)
        assert [f.label for f in kept] == ["Issuance"] and len(struck) == 1

    def test_normalise_keeps_meaning(self):
        assert normalise("  EGP 250 ") == normalise("EGP 250")
        assert normalise("EGP 250") != normalise("EGP 251")


class TestExtractFacts:
    def test_facts_the_page_supports_survive(self):
        llm = FakeLLMClient(lambda p: reply([
            {"label": "Issuance fee", "value": "EGP 250"},
            {"label": "Interest", "value": "4% monthly"}]))
        result = extract_facts("what are the fees?", PAGE, llm)
        assert result.used and len(result.facts) == 2 and result.struck == []

    def test_an_inferred_figure_never_survives(self):
        """The whole point: a model asked to extract will sometimes infer."""
        llm = FakeLLMClient(lambda p: reply([
            {"label": "Issuance fee", "value": "EGP 250"},
            {"label": "Two-year cost", "value": "EGP 500"}]))   # 250 + 250
        result = extract_facts("what are the fees?", PAGE, llm)
        assert [f.value for f in result.facts] == ["EGP 250"]
        assert result.struck == [("Two-year cost", "EGP 500")]

    def test_a_page_that_does_not_answer_returns_nothing(self):
        llm = FakeLLMClient(lambda p: reply([], present=False, note="not on this page"))
        result = extract_facts("what is the mortgage rate?", PAGE, llm)
        assert not result.used and result.present is False
        assert "does not answer" in result.reason

    def test_the_label_may_be_the_models_own_words(self):
        """This is how "Fees and Rates" on a page bridges to "fees and charges"."""
        llm = FakeLLMClient(lambda p: reply([{"label": "annual charge", "value": "EGP 250"}]))
        result = extract_facts("what are the charges?", PAGE, llm)
        assert result.facts[0].label == "annual charge"

    @pytest.mark.parametrize("bad", ["not json", '{"facts": "nope"}', "", "{}"])
    def test_malformed_output_changes_nothing(self, bad: str):
        result = extract_facts("q", PAGE, FakeLLMClient(lambda p: bad))
        assert not result.used

    def test_an_unreachable_model_changes_nothing(self):
        def dies(_p):
            raise LLMError("quota exhausted for today")
        result = extract_facts("q", PAGE, FakeLLMClient(dies))
        assert not result.used and "model unavailable" in result.reason

    def test_the_page_is_truncated_before_the_prompt(self):
        seen = {}
        llm = FakeLLMClient(lambda p: seen.setdefault("p", p) and reply([]) or reply([]))
        extract_facts("q", "x" * 50_000, llm, max_page_chars=1000)
        assert len(seen["p"]) < 5_000, "a 140k-character PDF must not become the prompt"


class TestTheLoggingTellsTheTwoFailuresApart:
    """The failure the user cannot detect without quota.

    "Found nothing on the page" is the page's answer and is often correct.
    "Found things and the formatting was rejected" is this system's problem.
    They look identical from outside, so they must not look identical in the log.
    """

    def test_nothing_found_is_logged_as_such(self, caplog):
        with caplog.at_level(logging.INFO, logger="agent.extraction"):
            extract_facts("q", PAGE, FakeLLMClient(lambda p: reply([], present=False)))
        assert "0 facts offered" in caplog.text
        assert "verbatim" not in caplog.text

    def test_everything_struck_is_logged_as_a_formatting_mismatch(self, caplog):
        with caplog.at_level(logging.INFO, logger="agent.extraction"):
            extract_facts("q", PAGE, FakeLLMClient(
                lambda p: reply([{"label": "a", "value": "250 EGP"},
                                 {"label": "b", "value": "EGP250"}])))
        assert "kept NONE" in caplog.text
        assert "formatting mismatch, not an empty page" in caplog.text
        assert "'250 EGP'" in caplog.text, "the rejected values must be shown"

    def test_a_partial_strike_reports_both_numbers(self, caplog):
        with caplog.at_level(logging.INFO, logger="agent.extraction"):
            extract_facts("q", PAGE, FakeLLMClient(
                lambda p: reply([{"label": "a", "value": "EGP 250"},
                                 {"label": "b", "value": "250 EGP"}])))
        assert "REJECTED 1 of 2" in caplog.text
        assert "KEPT 1 of 2" in caplog.text


class TestTheFallbackInTheLoop:
    """Once per sub-goal, at the outcome boundary, only after the keyword path failed."""

    def _run(self, extractor, task="What are the fees on the Classic credit card?",
             needles=("/Pages/Cards", "Credit%20Cards%20List", "Classic%20Credit%20Cards"),
             **kwargs):
        events: list[tuple[str, dict]] = []
        nav = choose_by(*needles)

        def respond(prompt: str) -> str:
            if PLANNING_MARKER in prompt:
                from conftest import echo_plan
                return echo_plan(prompt)
            if EXTRACTION_MARKER in prompt:
                return extractor(prompt)
            if "Write the answer" in prompt:
                return "x"
            return nav(prompt)

        loop = ResearchLoop(FakeLLMClient(respond), fetcher=live_fetcher(),
                            on_event=lambda n, d: events.append((n, d)),
                            compose=False, planning=False, fallback=True, **kwargs)
        return loop.run(task), events

    def test_it_costs_at_most_one_call_per_sub_goal(self):
        """Inside validate_fn this would have been one call per *page*."""
        calls = {"n": 0}

        def extractor(prompt: str) -> str:
            calls["n"] += 1
            return copy_from_page(prompt)

        result, _ = self._run(extractor)
        assert calls["n"] <= len(result.outcomes), calls["n"]

    def test_it_does_not_fire_when_the_keyword_path_produced_values(self):
        """Never speculative: a resolve backed by real values is left alone."""
        calls = {"n": 0}

        def extractor(prompt: str) -> str:
            calls["n"] += 1
            return copy_from_page(prompt)

        # The Classic card page yields three fee tables deterministically.
        self._run(extractor, task="What are the fees on the Classic credit card?")
        assert calls["n"] == 0, "the fallback fired on a page that already had values"

    def test_a_run_that_never_left_the_seed_does_not_pay_for_a_read(self):
        calls = {"n": 0}

        def extractor(prompt: str) -> str:
            calls["n"] += 1
            return copy_from_page(prompt)

        self._run(extractor, task="what about mortgages in Antarctica?", needles=("__none__",))
        assert calls["n"] == 0

    def test_facts_become_verified_claims(self):
        """The synthetic table flows through the existing claim pipeline."""
        result, _ = self._run(lambda p: copy_from_page(p, count=2),
                              task="How do I open an Islamic account?",
                              needles=("Accounts%20And%20Deposits",))
        if result.outcomes[0].extraction and result.outcomes[0].extraction["kept"]:
            assert result.claims_total > 0
            assert result.claims_supported == result.claims_total, (
                "a fact copied from the page must pass the attribution text check")

    def test_an_invented_value_never_reaches_a_claim(self):
        result, _ = self._run(lambda p: copy_from_page(p, count=1, plus_invented=True),
                              task="How do I open an Islamic account?",
                              needles=("Accounts%20And%20Deposits",))
        extraction = result.outcomes[0].extraction or {}
        assert any("99999" in s["value"] for s in extraction.get("struck", []))
        assert "99999" not in (result.answer or "")

    def test_the_fallback_is_reported_to_the_client(self):
        _result, events = self._run(lambda p: copy_from_page(p),
                                    task="How do I open an Islamic account?",
                                    needles=("Accounts%20And%20Deposits",))
        assert [d for n, d in events if n == "extraction"], "the read was not streamed"

    def test_off_by_flag_changes_nothing(self):
        calls = {"n": 0}

        def extractor(prompt: str) -> str:
            calls["n"] += 1
            return copy_from_page(prompt)

        events: list = []
        nav = choose_by("Accounts%20And%20Deposits")
        loop = ResearchLoop(
            FakeLLMClient(lambda p: extractor(p) if EXTRACTION_MARKER in p else nav(p)),
            fetcher=live_fetcher(), on_event=lambda n, d: events.append((n, d)),
            compose=False, planning=False, fallback=False)
        loop.run("How do I open an Islamic account?")
        assert calls["n"] == 0
