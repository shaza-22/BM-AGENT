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


class TestItFiresOnEveryRouteToAnEmptyAnswer:
    """Reported from a live run: nothing fired on ``arrived``.

        selection hop=3 outcome=arrived
        navigation arrived ... after 2 hops / 3 pages
        loop finished: sub_goals=1 resolved=0 ... answer=template

    18ms, no extraction line — on a page holding 23,710 characters of fee
    text. ``arrived`` is the navigator reaching a page that nothing validated,
    which is exactly what the fallback exists for and the most common way a run
    ends with no answer.

    The trigger was a whitelist of terminal statuses. A whitelist of the ways to
    fail is a list that will be incomplete again, so the question is now asked
    the other way round: are there facts? If not, read the page.
    """

    def _outcome(self, status: str, verdict=None):
        from agent.loop import SubGoalOutcome

        return SubGoalOutcome(sub_goal_id="sg_001", question="q",
                              status=status, verdict=verdict)

    @pytest.mark.parametrize("status", [
        "not_available",   # what `arrived` becomes
        "unresolved",
        "partial",
        "unreadable",      # was missed by the whitelist
        "something_added_later",
    ])
    def test_any_status_without_facts_triggers_it(self, status: str):
        assert ResearchLoop._needs_extraction(self._outcome(status)) is True

    def test_a_resolve_carrying_only_names_triggers_it_when_a_field_was_asked_for(self):
        """The green tick above "No verified facts were retrieved".

        A list of product names answers nothing when the sub-goal asked for
        fees, so the page gets re-read.
        """
        verdict = {"extracted": {"entities": [{"name": "Credit Card"}], "tables": []}}

        class _SG:
            target_fields = ["fees"]

        assert ResearchLoop._needs_extraction(self._outcome("resolved", verdict), _SG()) is True

    def test_a_list_of_names_is_the_answer_when_no_field_was_asked_for(self):
        """"What card types do you have" is answered by the names themselves.

        Re-reading there spends a call to be told what is already known, and
        this is the most common page shape on the site.
        """
        verdict = {"extracted": {"entities": [{"name": "Classic Credit Card"}], "tables": []}}

        class _SG:
            target_fields = ["overview"]

        assert ResearchLoop._needs_extraction(self._outcome("resolved", verdict), _SG()) is False

    def test_a_resolve_carrying_real_values_does_not(self):
        """Never speculative: a second opinion there can only agree."""
        verdict = {"extracted": {"tables": [{"records": [{"Issuance": "EGP 250"}]}]}}
        assert ResearchLoop._needs_extraction(self._outcome("resolved", verdict)) is False

    def test_it_fires_end_to_end_when_navigation_arrives(self):
        """The reported case, driven through the whole loop."""
        seen = {"extraction": 0}

        nav = choose_by("/Pages/Cards", "Credit%20Cards%20List")
        hops = {"n": 0}

        def respond(prompt: str) -> str:
            if PLANNING_MARKER in prompt:
                from conftest import echo_plan
                return echo_plan(prompt)
            if EXTRACTION_MARKER in prompt:
                seen["extraction"] += 1
                return copy_from_page(prompt)
            if "Write the answer" in prompt:
                return "x"
            hops["n"] += 1
            if hops["n"] >= 3:      # the selector declares the route ends here
                return json.dumps({"choice": -1, "outcome": "arrived",
                                   "reasoning": "this page is the destination",
                                   "confidence": 0.8})
            return nav(prompt)

        loop = ResearchLoop(FakeLLMClient(respond), fetcher=live_fetcher(),
                            on_event=lambda n, d: None, compose=False,
                            planning=False, fallback=True)
        result = loop.run("What are the fees on the Classic credit card?")

        assert result.outcomes[0].nav_status == "arrived"
        assert seen["extraction"] == 1, "the fallback did not fire on arrived"
        assert result.outcomes[0].extraction is not None

    def test_the_page_text_comes_from_the_result_not_a_side_effect(self):
        """``self._page_text`` is filled by validate_fn, which the navigator
        skips for a page it could not fetch. Depending on it made the fallback
        depend on a side effect that does not always happen -- and when it did
        not, the whole thing returned silently."""
        loop = ResearchLoop(FakeLLMClient(lambda p: "{}"), fetcher=live_fetcher(),
                            on_event=lambda n, d: None)

        class _Nav:
            page = {"url": "https://www.banquemisr.com/x", "text": "Issuance EGP 250"}
            trail: list = []

        url, text = loop._best_page_for(_Nav(), None)
        assert url.endswith("/x") and text == "Issuance EGP 250"


class TestEverySkipSaysWhy:
    """A silent skip is how this went unnoticed for a whole run."""

    def _loop(self, **kwargs):
        return ResearchLoop(FakeLLMClient(lambda p: "{}"), fetcher=live_fetcher(),
                            on_event=lambda n, d: None, fallback=True, **kwargs)

    def _nav(self, status="arrived", pages=3, page=None):
        class _Nav:
            pass

        nav = _Nav()
        nav.status, nav.pages_fetched, nav.trail = status, pages, []
        nav.page = page
        return nav

    def _sub_goal(self):
        class _SG:
            id = "sg_001"
            question = "q"
        return _SG()

    def test_nothing_to_read_is_logged(self, caplog):
        from agent.loop import SubGoalOutcome

        outcome = SubGoalOutcome(sub_goal_id="sg_001", question="q", status="not_available")
        with caplog.at_level(logging.INFO, logger="agent.loop"):
            self._loop()._fallback_extract(self._sub_goal(), self._nav(page=None), None, outcome)
        assert "nothing to read" in caplog.text and "arrived" in caplog.text

    def test_already_holding_facts_is_logged(self, caplog):
        from agent.loop import SubGoalOutcome

        outcome = SubGoalOutcome(
            sub_goal_id="sg_001", question="q", status="resolved",
            verdict={"extracted": {"tables": [{"records": [{"a": "b"}]}]}})
        with caplog.at_level(logging.INFO, logger="agent.loop"):
            self._loop()._fallback_extract(self._sub_goal(), self._nav(), None, outcome)
        assert "already holds facts" in caplog.text

    def test_a_blocked_run_is_logged_and_costs_nothing(self, caplog):
        from agent.loop import SubGoalOutcome

        outcome = SubGoalOutcome(sub_goal_id="sg_001", question="q", status="not_available")
        with caplog.at_level(logging.INFO, logger="agent.loop"):
            self._loop()._fallback_extract(
                self._sub_goal(), self._nav(status="blocked"), None, outcome)
        assert "blocked" in caplog.text

    def test_firing_announces_the_page_and_its_size(self, caplog):
        """So "it fired and found nothing" is distinguishable from "it never ran"."""
        from agent.loop import SubGoalOutcome

        outcome = SubGoalOutcome(sub_goal_id="sg_001", question="q", status="not_available")
        loop = ResearchLoop(
            FakeLLMClient(lambda p: json.dumps({"facts": [], "present": False, "note": ""})),
            fetcher=live_fetcher(), on_event=lambda n, d: None, fallback=True)
        nav = self._nav(page={"url": "https://www.banquemisr.com/fees.pdf",
                              "text": "Issuance EGP 250 " * 100})
        with caplog.at_level(logging.INFO, logger="agent.loop"):
            loop._fallback_extract(self._sub_goal(), nav, None, outcome)
        assert "firing for sg_001" in caplog.text
        assert "fees.pdf" in caplog.text and "chars" in caplog.text


class TestTheListPageWithoutItsMarker:
    """Traced from a live report: a list page returning "no verified facts".

    The vendored entity extractor depends on the literal line "More Details"
    following each product name. Measured on the credit-cards list fixture:

        with "More Details"    -> 12 entities -> validate() resolves
        without it             ->  0 entities -> validate() returns unresolved,
                                  "No relevant structured evidence found on page"

    So the answer went to zero at **stage 1**: not extraction finding nothing
    after validate accepted the page, and not the loop losing facts between
    extraction and the answer -- validate rejected the page outright, because
    the extractor saw no entities on a page that is nothing but a list of them.

    That is precisely the case the fallback exists for, and this pins that it
    rescues it.
    """

    FIXTURE = ("fixtures/live/home-smes-retail-banking-pages-cards-"
               "credit-cards-list.txt")
    URL = ("https://www.banquemisr.com/Home/SMEs/Retail%20Banking/Pages/"
           "Cards/Credit%20Cards%20List")

    def _pages(self):
        raw = (pathlib.Path(__file__).resolve().parent.parent / self.FIXTURE).read_text(
            encoding="utf-8", errors="replace")
        without = "\n".join(l for l in raw.splitlines() if l.strip() != "More Details")
        return raw, without

    def test_the_marker_is_what_the_extractor_depends_on(self):
        from person_b.extraction.extractor import extract_content

        raw, without = self._pages()
        assert len(extract_content(raw, ["overview"]).extracted.get("entities") or []) >= 8
        assert extract_content(without, ["overview"]).extracted.get("entities") == []

    def test_stage_one_is_where_it_goes_to_zero(self):
        from person_b.api import next_pending_sub_goal, plan_task, validate

        _raw, without = self._pages()
        sub_goal = next_pending_sub_goal(plan_task("what credit card types do you have"))
        verdict = validate(sub_goal, without, source_url=self.URL)
        assert verdict["resolved"] is False
        assert (verdict.get("extracted") or {}).get("entities") in (None, [], {})

    def test_the_fallback_rescues_exactly_this_page(self):
        from agent.loop import SubGoalOutcome, _Verdicts
        from person_b.api import next_pending_sub_goal, plan_task

        _raw, without = self._pages()
        page = {"url": self.URL, "text": without, "content_type": "html",
                "status": 200, "ok": True}

        def extractor(prompt: str) -> str:
            body = prompt.split("---", 1)[1].rsplit("---", 1)[0]
            names = [l.strip() for l in body.splitlines()
                     if "Credit Card" in l and len(l.strip()) < 60][:5]
            return json.dumps({"present": bool(names), "note": "",
                               "facts": [{"label": "Card type", "value": n} for n in names]})

        loop = ResearchLoop(
            FakeLLMClient(lambda p: extractor(p) if EXTRACTION_MARKER in p else "{}"),
            fetcher=live_fetcher(), on_event=lambda n, d: None,
            compose=False, planning=False, fallback=True)
        sub_goal = next_pending_sub_goal(plan_task("what credit card types do you have"))

        verdict = loop._make_validate_fn(sub_goal, _Verdicts())(sub_goal.question, page)
        assert verdict["resolved"] is False, "stage 1 should reject this page"

        outcome = SubGoalOutcome(sub_goal_id=sub_goal.id, question=sub_goal.question,
                                 status="not_available", verdict=verdict,
                                 nav_status="arrived", source_url=self.URL)

        class _Nav:
            status, pages_fetched, trail = "arrived", 3, []
            page = {"url": TestTheListPageWithoutItsMarker.URL, "text": without}

        loop._fallback_extract(sub_goal, _Nav(), None, outcome)

        assert outcome.status == "resolved", "the fallback did not rescue the page"
        assert outcome.extraction["kept"] >= 3
        assert any("Credit Card" in f["value"] for f in outcome.extraction["facts"])


class TestThreePageShapes:
    """The fallback as the general rescue, across the shapes the site has.

    Deterministic matching runs first everywhere -- fast, free, and reliable
    where it works. The fallback fires only where it found nothing. These are
    the three shapes that matter on this site, and the flag was re-enabled on
    the strength of them.
    """

    MULTI_TABLE = (
        "Accounts And Deposits\n"
        "- There are established limits on daily and monthly transactions, as well as "
        "on account balances, as detailed below:\n"
        "Amount | Limit\n"
        "Equivalent to 90,000 EGP | Maximum Daily Debit Transaction Limit\n"
        "Equivalent to 250,000 EGP | Maximum Monthly Debit Transaction Limit\n"
        "Equivalent to 1,000,000 EGP | Maximum Account Balance\n"
        "Currencies and Exchange Rates\n"
        "Buying | Cash\nTransfer | {{currencyCalculator.CashBuying}}\n"
    )

    def _claims_for(self, task: str, text: str, fallback: bool = True):
        from person_b.api import next_pending_sub_goal, plan_task, synthesize, validate_answer

        from agent.loop import SubGoalOutcome, _Verdicts

        url = "https://www.banquemisr.com/probe"
        page = {"url": url, "text": text, "content_type": "html", "status": 200, "ok": True}

        def extractor(prompt: str) -> str:
            body = prompt.split("---", 1)[1].rsplit("---", 1)[0]
            lines = [l.strip() for l in body.splitlines()
                     if 8 < len(l.strip()) < 70 and "|" not in l][:5]
            return json.dumps({"present": bool(lines), "note": "",
                               "facts": [{"label": f"Item {i + 1}", "value": l}
                                         for i, l in enumerate(lines)]})

        loop = ResearchLoop(
            FakeLLMClient(lambda p: extractor(p) if EXTRACTION_MARKER in p else "{}"),
            fetcher=live_fetcher(), on_event=lambda n, d: None,
            compose=False, planning=False, fallback=fallback)
        sub_goal = next_pending_sub_goal(plan_task(task))
        verdict = loop._make_validate_fn(sub_goal, _Verdicts())(sub_goal.question, page)
        outcome = SubGoalOutcome(
            sub_goal_id=sub_goal.id, question=sub_goal.question,
            status="resolved" if verdict["resolved"] else "not_available",
            verdict=verdict, nav_status="arrived", source_url=url)

        class _Nav:
            status, pages_fetched, trail = "arrived", 2, []

        _Nav.page = page
        loop._fallback_extract(sub_goal, _Nav(), None, outcome)

        verdicts = [outcome.verdict] if outcome.status == "resolved" else []
        synthesis = synthesize(task, validated_results=verdicts)
        verification = validate_answer(synthesis["claims"], [{"url": url, "text": text}],
                                       strict=True)
        return outcome, synthesis["claims"], verification

    def _page(self, stem: str, strip_marker: bool = False) -> str:
        raw = (pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "live"
               / f"{stem}.txt").read_text(encoding="utf-8", errors="replace")
        if strip_marker:
            raw = "\n".join(l for l in raw.splitlines() if l.strip() != "More Details")
        return raw

    def test_shape_one_fee_table_resolves_deterministically(self):
        """Deterministic wins here, so the fallback must not spend a call."""
        outcome, claims, verification = self._claims_for(
            "What are the fees on the Classic credit card?",
            self._page("home-smes-retail-banking-pages-cards-credit-cards-pages-"
                       "classic-credit-cards"))
        assert outcome.status == "resolved"
        assert outcome.extraction is None, "the fallback fired on a page that already had values"
        assert len(claims) > 20 and len(verification["passed"]) == len(claims)

    def test_shape_two_list_of_names_is_rescued_by_the_fallback(self):
        """The traced failure: deterministic sees nothing, the model reads it."""
        outcome, claims, verification = self._claims_for(
            "what credit card types do you have",
            self._page("home-smes-retail-banking-pages-cards-credit-cards-list",
                       strip_marker=True))
        assert outcome.extraction is not None and outcome.extraction["kept"] > 0
        assert outcome.status == "resolved"
        assert claims and len(verification["passed"]) == len(claims)

    def test_shape_three_multi_table_reads_correctly(self):
        """No repeated lead-in, columns the right way round, no markup."""
        _outcome, claims, verification = self._claims_for(
            "What are the account transaction limits?", self.MULTI_TABLE)
        statements = [c["statement"] for c in claims]

        assert len(statements) == len(set(statements)), f"repeated statements: {statements}"
        assert not any("established limits on daily and monthly" in s for s in statements), (
            "a sentence-length table name is being repeated in front of every row")
        assert any(s.startswith("Maximum Daily Debit Transaction Limit") for s in statements), (
            "the label column is still being taken by position, so rows read backwards")
        assert not any("{{" in s for s in statements), "template markup reached a claim"
        assert len(verification["passed"]) == len(claims)
