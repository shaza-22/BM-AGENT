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
from agent import config  # noqa: E402
from agent.loop import ResearchLoop, _Verdicts  # noqa: E402

CARDS = "/Pages/Cards"
CARD_LIST = "Credit%20Cards%20List"
# A task whose answer genuinely needs several entities looked at separately, so
# the expansion gate lets it through. A plain lookup no longer expands, which
# is the point of TestExpansionGate below.
EXPANDING_TASK = "Which credit card is best for travel?"
LOOKUP_TASK = "What are the fees on the Classic credit card?"
BOILERPLATE = "Retail Banking Corporate Banking Islamic Banking Quick Links E-Statement"


def build(*needles: str, **kwargs) -> tuple[ResearchLoop, list]:
    events: list[tuple[str, dict]] = []
    kwargs.setdefault("compose", False)    # navigation under test, not wording
    kwargs.setdefault("planning", False)   # ...nor decomposition: see TestModelPlanning
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
        loop.run(EXPANDING_TASK)
        expansions = [d for n, d in events if n == "plan_expanded"]
        assert expansions, "a recommendation task should expand the plan"
        assert expansions[0]["added"], "expansion reported nothing added"

    def test_sources_are_only_pages_the_run_actually_fetched(self):
        loop, _ = build(CARDS, CARD_LIST)
        result = loop.run("What credit cards does Banque Misr offer?")
        for url in result.source_urls:
            assert url in result.visited_urls, f"cited {url} without fetching it"


class TestBudgets:
    def test_the_page_budget_is_global_not_per_sub_goal(self):
        """The whole point of the global budget: caps must not multiply."""
        loop, events = build(CARDS, CARD_LIST, max_pages=2)
        result = loop.run(EXPANDING_TASK)
        assert result.pages_used <= 2 + 1, result.pages_used
        assert result.budget_exhausted

    def test_the_model_call_budget_stops_the_plan(self):
        # One call: enough for the first sub-goal to resolve and expand the
        # plan, nothing left for the sub-goal that expansion added. The budget
        # only trips when something is still pending -- a plan that simply runs
        # out of sub-goals has not been cut short and must not say it was.
        loop, _ = build(CARDS, CARD_LIST, max_llm_calls=1)
        result = loop.run(EXPANDING_TASK)
        assert result.llm_calls_used <= 2
        assert "model-call budget" in (result.budget_exhausted or "")

    def test_a_plan_that_simply_finishes_is_not_reported_as_cut_short(self):
        loop, _ = build(CARDS, CARD_LIST)
        result = loop.run(LOOKUP_TASK)
        assert result.budget_exhausted is None

    def test_sub_goals_dropped_for_budget_say_so(self):
        loop, events = build(CARDS, CARD_LIST, max_pages=2)
        result = loop.run(EXPANDING_TASK)
        dropped = [o for o in result.outcomes if o.status == "not_available"
                   and "budget" in o.reason]
        assert dropped, "the plan stopped for budget without naming the sub-goals it skipped"
        # And they reach the client, not just the result object.
        assert any(n == "sub_goal_finished" and "budget" in d.get("reason", "")
                   for n, d in events)

    def test_the_plan_never_exceeds_the_sub_goal_cap(self):
        loop, _ = build(CARDS, CARD_LIST, max_sub_goals=2)
        result = loop.run(EXPANDING_TASK)
        assert len(result.outcomes) <= 2

    def test_an_expanded_sub_goal_does_not_expand_again(self):
        loop, events = build(CARDS, CARD_LIST, max_expansion_depth=1)
        loop.run(EXPANDING_TASK)
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


class TestExpansionGate:
    """A task only fans out when answering it needs several entities.

    ``expand_plan`` reads the entity list off whatever page resolved and makes
    a sub-goal per entity, without consulting the task. Called on every resolve
    -- which is what this loop used to do -- a single-entity fee question
    became four sub-goals: the answer, then Gold, Platinum and Titanium, none
    of them asked for, spending the model budget and appearing under
    "Not found".

    The gate is the task type. This is the labelled set behind that choice; it
    is the measurement, so widen it rather than adjusting the gate by feel.
    """

    # (task, should_expand)
    CASES = [
        # Single-entity lookups. One page answers these.
        ("What are the fees on the Classic credit card?", False),
        ("What is the annual fee for the Classic credit card?", False),
        ("How much does the Gold card cost?", False),
        ("What is the grace period on the Classic card?", False),
        ("What is the interest rate on the Titanium card?", False),
        ("What is the daily ATM withdrawal limit?", False),
        # How-to and location questions.
        ("How do I open an Islamic account?", False),
        ("How do I activate my debit card at the ATM?", False),
        ("Where is the schedule of fees and commissions?", False),
        ("Where can I find the branch list?", False),
        # Open questions answered by one hub page.
        ("What credit cards does Banque Misr offer?", False),
        ("What personal loans are available?", False),
        ("Tell me about Banque Misr payment cards", False),
        ("What accounts and deposits does the bank offer?", False),
        # Comparisons.
        ("Compare the Classic and Gold credit cards", True),
        ("What is the difference between the Titanium and Platinum cards?", True),
        ("Classic vs Gold credit card", True),
        ("Which is better, a personal loan or a car loan?", True),
        # Recommendations.
        ("Which card is best for travel?", True),
        ("Recommend a card for online shopping", True),
        ("Which account should I get for my salary?", True),
        # Enumerate-and-detail.
        ("List all the credit cards and their annual fees", True),
        ("Show me every loan and its interest rate", True),
    ]

    @pytest.mark.parametrize("task,should_expand", CASES)
    def test_the_gate_matches_the_label(self, task: str, should_expand: bool) -> None:
        from person_b.api import plan_task

        plan = plan_task(task)
        gated = plan.goal.task_type in ResearchLoop.EXPANDING_TASK_TYPES
        assert gated is should_expand, (
            f"{task!r} classified {plan.goal.task_type!r}; "
            f"expansion would be {gated}, labelled {should_expand}"
        )

    def test_a_direct_lookup_produces_exactly_one_sub_goal(self):
        """The regression this whole gate exists for."""
        loop, _ = build(CARDS, CARD_LIST, "Classic%20Credit%20Cards")
        result = loop.run(LOOKUP_TASK)
        assert len(result.outcomes) == 1, [o.question for o in result.outcomes]
        assert result.budget_exhausted is None


class TestNamedEntityNarrowing:
    """Expansion follows the entities the task named, when it named any."""

    CARDS_FOUND = [
        "Classic Credit Card", "Gold Credit Card", "Platinum Visa - MasterCredit Card",
        "Titanium Credit Card", "Visa Infinite", "World Credit Card",
    ]
    LOANS_FOUND = ["Personal Loans", "Car Loans", "Mortgage Finance", "Payroll Loans"]

    def test_a_named_comparison_expands_only_to_those_entities(self):
        from agent.loop import _entities_named_by

        named = _entities_named_by("Compare the Classic and Gold credit cards", self.CARDS_FOUND)
        assert sorted(named) == ["Classic Credit Card", "Gold Credit Card"]

    def test_words_shared_by_most_candidates_do_not_single_anything_out(self):
        """"credit" and "card" are on nearly every candidate and decide nothing.

        The stop set is derived from the candidates each run rather than
        written down, which is what keeps this free of category vocabulary --
        and why it behaves identically on loans.
        """
        from agent.loop import _entities_named_by

        assert _entities_named_by("Which credit card is best?", self.CARDS_FOUND) == []
        assert sorted(_entities_named_by("Compare car loans and personal loans",
                                         self.LOANS_FOUND)) == ["Car Loans", "Personal Loans"]

    def test_one_outlier_name_cannot_defeat_the_stop_set(self):
        """"Visa Infinite" omits "credit"/"card"; majority presence, not all."""
        from agent.loop import _entities_named_by

        assert "Titanium Credit Card" not in _entities_named_by(
            "Compare the Classic and Gold credit cards", self.CARDS_FOUND)

    def test_an_open_task_still_expands_to_everything(self):
        from agent.loop import ResearchLoop

        verdict = {"resolved": True,
                   "extracted": {"entities": [{"name": n} for n in self.CARDS_FOUND]}}
        same = ResearchLoop._narrow_to_named_entities(verdict, "Which card is best for travel?")
        assert len(same["extracted"]["entities"]) == len(self.CARDS_FOUND)


class TestModelPlanning:
    """Decomposition by the model, and the ways it must fail safe.

    The vendored planner decomposes by keyword and produced exactly one
    sub-goal for every task measured, so a plan panel showed one restated line.
    This replaces the questions; it must never be able to leave a run without a
    plan.
    """

    def _run(self, respond, task, **kwargs):
        events: list[tuple[str, dict]] = []
        loop = ResearchLoop(FakeLLMClient(respond), fetcher=live_fetcher(),
                            on_event=lambda n, d: events.append((n, d)),
                            compose=False, planning=True, **kwargs)
        return loop.run(task), events

    def _script(self, plan_json: str, *needles: str):
        """Answer the planning prompt with plan_json, navigate by needles."""
        from conftest import PLANNING_MARKER

        nav = choose_by(*needles)
        return lambda p: plan_json if PLANNING_MARKER in p else nav(p)

    def test_a_decomposed_plan_replaces_the_keyword_one(self):
        from conftest import plan_reply

        plan = plan_reply("Find the list of Banque Misr credit cards",
                          "Find the fees of the Classic credit card",
                          reasoning="the request needs the list, then the fees")
        result, events = self._run(
            self._script(plan, CARDS, CARD_LIST, "Classic%20Credit%20Cards"),
            "What credit cards are there and what do they cost?")

        assert result.plan_source == "model"
        assert result.plan_reasoning == "the request needs the list, then the fees"
        assert len(result.outcomes) == 2
        assert [o.question for o in result.outcomes] == [
            "Find the list of Banque Misr credit cards",
            "Find the fees of the Classic credit card",
        ]

    def test_the_plan_event_carries_its_provenance_and_rationale(self):
        from conftest import plan_reply

        _r, events = self._run(
            self._script(plan_reply("Find the list of Banque Misr credit cards"), CARDS),
            "What credit cards does Banque Misr offer?")
        plan_event = next(d for n, d in events if n == "plan")
        assert plan_event["plan_source"] == "model"
        assert plan_event["sub_goals"][0]["why"], "the per-sub-goal rationale was dropped"

    def test_planning_costs_exactly_one_call(self):
        from conftest import plan_reply

        with_planning, _ = self._run(
            self._script(plan_reply("Find the list of Banque Misr credit cards"), CARDS, CARD_LIST),
            "What credit cards does Banque Misr offer?")
        without = ResearchLoop(FakeLLMClient(choose_by(CARDS, CARD_LIST)),
                               fetcher=live_fetcher(), on_event=lambda n, d: None,
                               compose=False, planning=False,
                               ).run("What credit cards does Banque Misr offer?")
        assert with_planning.llm_calls_used == without.llm_calls_used + 1

    def test_the_cap_is_enforced_on_what_the_model_returns(self):
        from conftest import plan_reply

        result, _ = self._run(
            self._script(plan_reply(*[f"Find thing number {i}" for i in range(9)]), CARDS),
            "do everything", max_sub_goals=9)
        assert len(result.outcomes) <= config.MAX_PLANNED_SUB_GOALS

    # --- failing safe ----------------------------------------------------
    @pytest.mark.parametrize("reply,why", [
        ("not json at all", "unparseable"),
        ('{"sub_goals": []}', "empty list"),
        ('{"sub_goals": [{"question": "", "why": "x"}]}', "blank question"),
        ('{"sub_goals": [{"question": "short"}]}', "too short to navigate"),
        ('{"nope": 1}', "no sub_goals key"),
    ])
    def test_a_bad_plan_falls_back_to_the_keyword_planner(self, reply: str, why: str):
        result, _ = self._run(self._script(reply, CARDS, CARD_LIST),
                              "What credit cards does Banque Misr offer?")
        assert result.plan_source == "keyword", why
        assert result.outcomes, "the run lost its plan entirely"

    def test_an_unreachable_model_still_leaves_a_plan(self):
        from agent.llm import LLMError
        from conftest import PLANNING_MARKER

        nav = choose_by(CARDS, CARD_LIST)

        def respond(prompt: str) -> str:
            if PLANNING_MARKER in prompt:
                raise LLMError("quota exhausted for today")
            return nav(prompt)

        result, _ = self._run(respond, "What credit cards does Banque Misr offer?")
        assert result.plan_source == "keyword"
        assert result.outcomes

    def test_duplicate_sub_goals_are_dropped(self):
        from agent.planner import parse_plan

        parsed = parse_plan(
            '{"sub_goals": [{"question": "Find the credit card list"},'
            ' {"question": "find the CREDIT card list"},'
            ' {"question": "Find the fees of the Classic card"}], "reasoning": ""}',
            max_sub_goals=3)
        assert len(parsed) == 2

    # --- the conflict with expansion -------------------------------------
    def test_a_decomposed_plan_turns_expansion_off(self):
        """Two mechanisms doing one job would double-count and duplicate."""
        from conftest import plan_reply

        _result, events = self._run(
            self._script(plan_reply("Find the list of Banque Misr credit cards",
                                    "Find the fees of the Classic credit card"),
                         CARDS, CARD_LIST, "Classic%20Credit%20Cards"),
            "Which credit card is best for travel?")
        assert not [n for n, _d in events if n == "plan_expanded"]

    def test_a_single_sub_goal_plan_leaves_expansion_on(self):
        """Expansion is the fallback for entities nothing could know up front."""
        from conftest import plan_reply

        _result, events = self._run(
            self._script(plan_reply("Find the list of Banque Misr credit cards"),
                         CARDS, CARD_LIST),
            "Which credit card is best for travel?")
        assert [n for n, _d in events if n == "plan_expanded"]

    def test_off_by_config_is_a_one_line_revert(self):
        result = ResearchLoop(FakeLLMClient(choose_by(CARDS)), fetcher=live_fetcher(),
                              on_event=lambda n, d: None, compose=False, planning=False,
                              ).run("What credit cards does Banque Misr offer?")
        assert result.plan_source == "keyword"
