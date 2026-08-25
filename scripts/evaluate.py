"""
End-to-end evaluation of the agent, offline, one case per requirement category.

What it does
    Runs the whole pipeline -- plan, navigate from the homepage, validate,
    extract, verify, answer -- against the saved pages, and reports pass or
    fail per case.

Run
    python3 scripts/evaluate.py            # all categories
    python3 scripts/evaluate.py lookup     # one category

Why this exists, and what the other two harnesses are not
    ``scripts/validator_bench.py`` measures one function, ``validate()``, on 26
    labelled (task, page) pairs. It never navigates, never runs the loop, never
    reaches an answer. Person B's ``PersonBEvaluator`` loads fixtures directly
    and tests their layer in isolation. Both are useful and neither evaluates
    *the agent*: a run can navigate perfectly and still answer nothing, which
    is precisely the class of failure this project has hit repeatedly.

What it honestly tests, and what it does not
    The model is a scripted stand-in, so this measures **the pipeline**, not the
    quality of a real model's judgement. It answers "does a task of this shape
    reach a grounded answer, and does a question the site cannot answer come
    back as not-found", which no other harness here answers. It cannot tell you
    whether a live model picks good links or writes good analysis.

    The navigation script per case names link *labels*, not indexes, so a
    change to ranking weights does not silently rewrite what is being tested.
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

from conftest import EXTRACTION_MARKER, PLANNING_MARKER, choose_by, live_fetcher  # noqa: E402

from agent.answer import COMPOSING_MARKER  # noqa: E402
from agent.llm import FakeLLMClient  # noqa: E402
from agent.loop import LoopResult, ResearchLoop  # noqa: E402
from agent.session import RESOLVER_MARKER, Session, resolve_task  # noqa: E402


@dataclass
class Case:
    name: str
    category: str
    task: str
    route: tuple[str, ...]                 # link labels to follow, in order
    expect: Callable[[LoopResult], tuple[bool, str]]
    follow_up: str | None = None           # a second turn in the same session
    # The follow-up walks from the homepage again -- every turn does -- so it
    # needs its own route. Sharing one script across both turns left the second
    # with nothing to follow, which is a harness bug that reads exactly like a
    # navigation failure.
    follow_up_route: tuple[str, ...] = ()
    plan: list[str] | None = None          # sub-goals the planner should return
    facts_from_page: bool = True           # let the extraction fallback read


# --- expectations ---------------------------------------------------------
def answered_with_sources(result: LoopResult) -> tuple[bool, str]:
    if not result.resolved_count:
        return False, f"nothing resolved (nav ended {result.outcomes[0].nav_status})"
    if not result.claims_total:
        return False, "resolved but produced no claims"
    if result.claims_supported != result.claims_total:
        return False, f"only {result.claims_supported}/{result.claims_total} claims supported"
    if not result.source_urls:
        return False, "an answer with no sources"
    return True, f"{result.claims_total} claims, all supported, {len(result.source_urls)} source(s)"


def reported_not_found(result: LoopResult) -> tuple[bool, str]:
    """The invariant that must survive every other fix: a correct no stays no."""
    if result.resolved_count:
        return False, "resolved a question the site cannot answer"
    if result.claims_total:
        return False, f"produced {result.claims_total} claims for an unanswerable question"
    if result.answer_source != "template":
        return False, "generated prose with nothing to ground on"
    return True, "no claims, no generated prose, honest not-found"


def several_sub_goals(result: LoopResult) -> tuple[bool, str]:
    if len(result.outcomes) < 2:
        return False, f"decomposed into only {len(result.outcomes)} sub-goal(s)"
    if result.plan_source != "model":
        return False, "the plan came from the keyword planner"
    ok, detail = answered_with_sources(result)
    return ok, f"{len(result.outcomes)} sub-goals; {detail}"


CASES: list[Case] = [
    Case(
        name="single fee lookup",
        category="lookup",
        task="What are the fees on the Classic credit card?",
        route=("/Pages/Cards", "Credit%20Cards%20List", "Classic%20Credit%20Cards"),
        expect=answered_with_sources,
    ),
    Case(
        name="list of products",
        category="list",
        task="what credit card types do you have",
        route=("/Pages/Cards", "Credit%20Cards%20List"),
        expect=answered_with_sources,
    ),
    Case(
        name="comparison across two entities",
        category="comparison",
        task="Compare the Classic and Gold credit cards",
        route=("/Pages/Cards", "Credit%20Cards%20List", "Classic%20Credit%20Cards"),
        plan=["Find the fees of the Classic credit card",
              "Find the fees of the Gold credit card"],
        expect=several_sub_goals,
    ),
    Case(
        name="unanswerable question",
        category="missing-information trap",
        task="Can I open a joint account with my dog?",
        route=("Accounts%20And%20Deposits",),
        expect=reported_not_found,
        facts_from_page=False,      # the page genuinely does not answer it
    ),
    Case(
        name="follow-up using conversation context",
        category="context",
        task="What credit cards does Banque Misr offer?",
        follow_up="and what are the fees on the first one?",
        route=("/Pages/Cards", "Credit%20Cards%20List"),
        follow_up_route=("/Pages/Cards", "Credit%20Cards%20List", "Classic%20Credit%20Cards"),
        expect=answered_with_sources,
    ),
]


# --- running --------------------------------------------------------------
def responder(case: Case, route: tuple[str, ...] | None = None) -> Callable[[str], str]:
    """One scripted stand-in for the model, covering every prompt the loop sends."""
    navigate = choose_by(*(route if route is not None else case.route))

    def respond(prompt: str) -> str:
        if PLANNING_MARKER in prompt:
            questions = case.plan or [f"Find information for: {case.task}"]
            return json.dumps({
                "sub_goals": [{"question": q, "why": "needed for this case"} for q in questions],
                "reasoning": "scripted plan for the evaluation harness",
            })
        if EXTRACTION_MARKER in prompt:
            if not case.facts_from_page:
                return json.dumps({"facts": [], "present": False,
                                   "note": "this page does not answer the question"})
            body = prompt.split("---", 1)[1].rsplit("---", 1)[0]
            lines = [l.strip() for l in body.splitlines()
                     if 8 < len(l.strip()) < 70 and "|" not in l][:5]
            return json.dumps({"present": bool(lines), "note": "",
                               "facts": [{"label": f"Detail {i + 1}", "value": l}
                                         for i, l in enumerate(lines)]})
        if RESOLVER_MARKER in prompt:
            # The context requirement: rewrite the follow-up into something
            # readable on its own. Scripted, so the harness tests the wiring
            # rather than a model's phrasing.
            return json.dumps({
                "sub_goal": "What are the fees on the Classic credit card?",
                "used_context": True,
                "reasoning": "the previous turn listed the cards; 'the first one' is the Classic",
            })
        if COMPOSING_MARKER in prompt:
            # Composition is exercised elsewhere; here it must not decide the
            # verdict, so it declines and the template answer stands.
            return json.dumps({"sentences": []})
        return navigate(prompt)

    return respond


def run_case(case: Case) -> tuple[bool, str, LoopResult]:
    llm = FakeLLMClient(responder(case))
    fetcher = live_fetcher()
    session = Session() if case.follow_up else None

    loop = ResearchLoop(llm, fetcher=fetcher, on_event=lambda n, d: None)
    result = loop.run(case.task)

    if case.follow_up:
        # The context requirement: the second turn must be readable only
        # against the first. resolve_task is what makes it so.
        session.add_turn(case.task, resolve_task(case.task, session, llm), result)
        resolution = resolve_task(case.follow_up, session, llm)
        if resolution.sub_goal == case.follow_up:
            return False, "the follow-up was not resolved against the conversation", result
        follow_llm = FakeLLMClient(responder(case, case.follow_up_route or case.route))
        loop = ResearchLoop(follow_llm, fetcher=fetcher, on_event=lambda n, d: None)
        result = loop.run(resolution.sub_goal)

    ok, detail = case.expect(result)
    return ok, detail, result


def main(argv: list[str]) -> int:
    # WARNING too: the composer declines by design in this harness and says
    # so loudly, which would otherwise be noise in the middle of the report.
    logging.disable(logging.WARNING)
    wanted = argv[1] if len(argv) > 1 else None
    cases = [c for c in CASES if not wanted or c.category == wanted]
    if not cases:
        print(f"no cases in category {wanted!r}. "
              f"Categories: {sorted({c.category for c in CASES})}")
        return 2

    print(f"{'category':26} {'case':38} {'result':6}  detail")
    print("-" * 108)
    passed = 0
    for case in cases:
        try:
            ok, detail, result = run_case(case)
        except Exception as exc:                       # a harness bug is a fail, loudly
            ok, detail = False, f"{type(exc).__name__}: {exc}"
            result = None
        passed += ok
        print(f"{case.category:26} {case.name:38} {'PASS' if ok else 'FAIL':6}  {detail[:52]}")
        if result is not None:
            print(f"{'':26} {'':38} {'':6}  "
                  f"{result.pages_used} pages, {result.llm_calls_used} model calls, "
                  f"answer from {result.answer_source}")

    print("-" * 108)
    print(f"{passed}/{len(cases)} passed")
    print("\nThis measures the pipeline, not a live model's judgement: the model is "
          "scripted.\nIt cannot tell you whether a real model picks good links or "
          "writes good analysis.")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
