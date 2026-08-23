"""The orchestrator: plan, budgets, the partial policy, and the acceptance gate.

Everything here is offline. Pages come from ``fixtures/live/`` through a
``FakeSession``, so the real Fetcher runs; the model is a scripted
``FakeLLMClient`` that picks links by label.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from conftest import choose_by, live_fetcher  # noqa: E402

from agent.acceptance import AcceptanceGate, gate_from_config  # noqa: E402
from agent.llm import FakeLLMClient, LLMError  # noqa: E402
from agent.loop import ResearchLoop, _Verdicts  # noqa: E402

CARDS = "/Pages/Cards"
CARD_LIST = "Credit%20Cards%20List"
BOILERPLATE = "Retail Banking Corporate Banking Islamic Banking Quick Links E-Statement"


def build(*needles: str, **kwargs) -> tuple[ResearchLoop, list]:
    events: list[tuple[str, dict]] = []
    loop = ResearchLoop(
        FakeLLMClient(choose_by(*needles)),
        fetcher=live_fetcher(),
        on_event=lambda name, data: events.append((name, data)),
        **kwargs,
    )
    return loop, events


def names(events) -> list[str]:
    return [name for name, _ in events]


class TestTheHappyPath:
    def test_a_task_becomes_a_plan_and_an_answer(self):
        loop, events = build(CARDS, CARD_LIST)
        result = loop.run("What credit cards does Banque Misr offer?")

        assert result.resolved_count >= 1
        assert result.answer
        assert result.source_urls, "an answer with no sources is unattributed"
        assert names(events)[0] == "plan"
        assert "answer_verified" in names(events)

    def test_every_sub_goal_is_reported_started_and_finished(self):
        """No sub-goal may disappear silently.

        Regression: ``apply_validation`` returns an *unresolved* sub-goal to
        PENDING, so the loop kept selecting the same one; guarding that by
        stopping dropped every sub-goal behind it without a word.
        """
        loop, events = build(CARDS, CARD_LIST)
        loop.run("What credit cards does Banque Misr offer?")

        started = [d["sub_goal_id"] for n, d in events if n == "sub_goal_started"]
        finished = [d["sub_goal_id"] for n, d in events if n == "sub_goal_finished"]
        assert started, "no sub-goal was started"
        assert set(started) <= set(finished), "a started sub-goal never finished"
        assert len(finished) == len(set(finished)), "a sub-goal finished twice"

    def test_expansion_is_reported_with_what_it_added(self):
        loop, events = build(CARDS, CARD_LIST)
        loop.run("What credit cards does Banque Misr offer?")
        expansions = [d for n, d in events if n == "plan_expanded"]
        assert expansions, "a resolved discovery sub-goal should expand the plan"
        assert expansions[0]["added"], "expansion reported nothing added"

    def test_sources_are_only_pages_the_run_actually_fetched(self):
        loop, _ = build(CARDS, CARD_LIST)
        result = loop.run("What credit cards does Banque Misr offer?")
        for url in result.source_urls:
            assert url in result.visited_urls, f"cited {url} without fetching it"


class TestBudgets:
    def test_the_page_budget_is_global_not_per_sub_goal(self):
        """The whole point of the global budget: caps must not multiply."""
        loop, events = build(CARDS, CARD_LIST, max_pages=3)
        result = loop.run("What credit cards does Banque Misr offer?")
        assert result.pages_used <= 3 + 1, result.pages_used
        assert result.budget_exhausted

    def test_the_model_call_budget_stops_the_plan(self):
        loop, _ = build(CARDS, CARD_LIST, max_llm_calls=2)
        result = loop.run("What credit cards does Banque Misr offer?")
        assert result.llm_calls_used <= 3
        assert "model-call budget" in (result.budget_exhausted or "")

    def test_sub_goals_dropped_for_budget_say_so(self):
        loop, events = build(CARDS, CARD_LIST, max_pages=3)
        result = loop.run("What credit cards does Banque Misr offer?")
        dropped = [o for o in result.outcomes if o.status == "not_available"
                   and "budget" in o.reason]
        assert dropped, "the plan stopped for budget without naming the sub-goals it skipped"
        # And they reach the client, not just the result object.
        assert any(n == "sub_goal_finished" and "budget" in d.get("reason", "")
                   for n, d in events)

    def test_the_plan_never_exceeds_the_sub_goal_cap(self):
        loop, _ = build(CARDS, CARD_LIST, max_sub_goals=2)
        result = loop.run("What credit cards does Banque Misr offer?")
        assert len(result.outcomes) <= 2

    def test_an_expanded_sub_goal_does_not_expand_again(self):
        loop, events = build(CARDS, CARD_LIST, max_expansion_depth=1)
        loop.run("What credit cards does Banque Misr offer?")
        parents = [d["parent_id"] for n, d in events if n == "plan_expanded"]
        assert len(set(parents)) <= 1, "expansion recursed past its depth cap"


class TestFailuresAreNotEmptyAnswers:
    def test_a_model_failure_is_an_error_not_a_shrug(self):
        """A spent quota and an unanswerable question must not look alike."""

        def dies(_prompt: str) -> str:
            raise LLMError("quota exhausted for today")

        loop = ResearchLoop(FakeLLMClient(dies), fetcher=live_fetcher(),
                            on_event=lambda n, d: None)
        result = loop.run("What credit cards does Banque Misr offer?")
        assert result.error_reason, "an unreachable model was reported as 'nothing found'"

    def test_one_model_failure_stops_the_plan(self):
        calls = {"n": 0}

        def dies(_prompt: str) -> str:
            calls["n"] += 1
            raise LLMError("quota exhausted for today")

        loop = ResearchLoop(FakeLLMClient(dies), fetcher=live_fetcher(),
                            on_event=lambda n, d: None)
        loop.run("What credit cards does Banque Misr offer?")
        # llm.py owns retries; the loop must not spend a page per sub-goal
        # rediscovering that the model is gone.
        assert calls["n"] <= 2, calls["n"]


class TestTheAcceptanceGate:
    """The gate is a second opinion, one-way, and it explains itself."""

    def _gate(self, pages: int = 3) -> AcceptanceGate:
        gate = gate_from_config()
        for i in range(pages):
            gate.observe(f"https://example.test/{i}", f"{BOILERPLATE} unique-body-{i} " + "x" * 50)
        return gate

    def test_evidence_repeated_across_pages_is_chrome(self):
        decision = self._gate().judge(
            {"resolved": True, "evidence": [{"evidence_text": BOILERPLATE}]},
            "https://example.test/new",
        )
        assert decision.accepted is False
        assert "chrome" in decision.reason

    def test_evidence_unique_to_the_page_passes(self):
        decision = self._gate().judge(
            {"resolved": True,
             "evidence": [{"evidence_text": "The issuance fee is 100 EGP annually."}]},
            "https://example.test/new",
        )
        assert decision.accepted is True

    def test_one_unique_snippet_is_enough(self):
        decision = self._gate().judge(
            {"resolved": True, "evidence": [
                {"evidence_text": BOILERPLATE},
                {"evidence_text": "The issuance fee is 100 EGP annually."},
            ]},
            "https://example.test/new",
        )
        assert decision.accepted is True
        assert decision.snippets_unique == 1

    def test_the_gate_never_promotes(self):
        """One-way precedence: it can withhold a resolve, never grant one."""
        verdict = {"resolved": False, "status": "unresolved",
                   "evidence": [{"evidence_text": "Something unique to this page entirely."}]}
        decision = self._gate().judge(verdict, "https://example.test/new")
        assert decision.accepted is True
        assert "nothing to gate" in decision.reason

    def test_it_abstains_early_in_a_run_and_says_so(self):
        gate = self._gate(pages=1)
        decision = gate.judge(
            {"resolved": True, "evidence": [{"evidence_text": BOILERPLATE}]},
            "https://example.test/new",
        )
        assert decision.accepted is True
        assert "too few to tell" in decision.reason

    def test_short_snippets_are_not_evidence_of_anything(self):
        decision = self._gate().judge(
            {"resolved": True, "evidence": [{"evidence_text": "Cash"}]},
            "https://example.test/new",
        )
        assert decision.accepted is True
        assert "no quotable evidence" in decision.reason

    def test_a_withheld_resolve_reaches_the_client_and_the_result(self):
        """A silent override is what makes a system impossible to debug."""
        gate = AcceptanceGate(enabled=True, min_repeats=0)  # reject everything gateable
        events: list[tuple[str, dict]] = []
        loop = ResearchLoop(
            FakeLLMClient(choose_by(CARDS, CARD_LIST)),
            fetcher=live_fetcher(), gate=gate,
            on_event=lambda n, d: events.append((n, d)),
        )
        result = loop.run("What credit cards does Banque Misr offer?")

        assert result.gate_rejections > 0, "min_repeats=0 should reject every resolve"
        gate_events = [d for n, d in events if n == "gate"]
        assert gate_events, "a withheld resolve was not streamed"
        assert gate_events[0]["accepted"] is False
        assert gate_events[0]["validator_reason"], "the validator's own reason was dropped"

    def test_a_withheld_resolve_keeps_navigating(self):
        gate = AcceptanceGate(enabled=True, min_repeats=0)
        loop = ResearchLoop(FakeLLMClient(choose_by(CARDS, CARD_LIST)),
                            fetcher=live_fetcher(), gate=gate, on_event=lambda n, d: None)
        result = loop.run("What credit cards does Banque Misr offer?")
        assert result.resolved_count == 0
        assert result.pages_used >= 2, "the run should have carried on past the withheld page"

    def test_disabled_the_gate_changes_nothing(self):
        gate = AcceptanceGate(enabled=False, min_repeats=0)
        decision = gate.judge({"resolved": True, "evidence": [{"evidence_text": BOILERPLATE}]}, "u")
        assert decision.accepted is True

    def test_it_ships_enabled(self):
        assert gate_from_config().enabled is True


class TestVerdictCollection:
    """``NavigationResult.extracted`` is extracted *data*, not the verdict."""

    def test_the_best_verdict_wins_regardless_of_arrival_order(self):
        collected = _Verdicts()
        collected.record("a", {"resolved": False, "status": "unresolved"})
        collected.record("b", {"resolved": False, "status": "partial"})
        collected.record("c", {"resolved": False, "status": "unresolved"})
        assert collected.best["status"] == "partial"

    def test_a_resolve_outranks_a_partial(self):
        collected = _Verdicts()
        collected.record("a", {"resolved": False, "status": "partial"})
        collected.record("b", {"resolved": True, "status": "resolved"})
        assert collected.best["resolved"] is True

    def test_a_partial_survives_a_run_that_never_resolved(self):
        """The partial policy depends on this: evidence already paid for."""
        collected = _Verdicts()
        collected.record("a", {"resolved": False, "status": "partial",
                               "reason": "found fees, missing benefits"})
        assert collected.best is not None
        assert collected.best["status"] == "partial"


class TestPartialPolicy:
    def test_partial_does_not_stop_navigation(self):
        from agent import config

        assert config.PARTIAL_STOPS_NAVIGATION is False, (
            "partial means the rest is one hop deeper; stopping there throws "
            "away the remaining hop budget"
        )
