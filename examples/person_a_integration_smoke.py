"""Runnable smoke test of the Person B contract, offline.

Person B's README and handoff both point at this file as the first thing to
run, but it was absent from the delivered package (see
``src/person_b/VENDORED.md``). It is written here rather than striking the
references, because a one-command check that the intelligence layer imports
and round-trips is worth having.

What it exercises, in the order the live loop calls them:

    plan_task -> next_pending_sub_goal -> validate -> apply_validation
              -> expand_plan -> synthesize -> validate_answer -> finalize

No network and no model: pages come from ``fixtures/live/``, which are real
pages saved from banquemisr.com. Exit status is 0 when every stage produced
the shape the loop depends on.

    python examples/person_a_integration_smoke.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from person_b.api import (  # noqa: E402
    apply_validation,
    expand_plan,
    finalize,
    next_pending_sub_goal,
    plan_task,
    synthesize,
    validate,
    validate_answer,
)

FIXTURES = ROOT / "fixtures" / "live"
TASK = "What credit cards does Banque Misr offer?"
CARD_LIST_PAGE = "home-smes-retail-banking-pages-cards-credit-cards-list"
CARD_LIST_URL = (
    "https://www.banquemisr.com/Home/SMEs/Retail-Banking/Pages/"
    "Cards/Credit-Cards/Credit-Cards-List"
)


def _page(stem: str) -> str:
    path = FIXTURES / f"{stem}.txt"
    if not path.exists():
        raise SystemExit(f"missing fixture: {path}  (run scripts/save_fixtures.py first)")
    return path.read_text(encoding="utf-8", errors="replace")


def _head(n: int, title: str) -> None:
    print(f"\n{n}. {title}\n{'-' * (len(title) + 3)}")


def main() -> int:
    failures: list[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        print(f"   [{'ok' if condition else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
        if not condition:
            failures.append(label)

    _head(1, "plan_task")
    plan = plan_task(TASK)
    sub_goal = next_pending_sub_goal(plan)
    print(f"   task       : {TASK}")
    print(f"   task_type  : {plan.goal.task_type}")
    print(f"   sub-goal   : {sub_goal.question}")
    print(f"   fields     : {sub_goal.target_fields}")
    check("plan has at least one pending sub-goal", sub_goal is not None)

    _head(2, "validate (the function Person A's navigator calls per page)")
    verdict = validate(sub_goal, _page(CARD_LIST_PAGE), source_url=CARD_LIST_URL)
    print(f"   resolved   : {verdict['resolved']}")
    print(f"   status     : {verdict.get('status')}")
    print(f"   reason     : {verdict.get('reason')}")
    check("verdict carries the three contract keys",
          all(k in verdict for k in ("resolved", "extracted", "reason")))
    check("the credit-cards list page resolves the discovery sub-goal", verdict["resolved"] is True)

    _head(3, "validate rejects a page that does not answer the task")
    off = validate(
        sub_goal,
        _page("home-smes-retail-banking-accounts-and-deposits"),
        source_url="https://www.banquemisr.com/Home/SMEs/Retail-Banking/Accounts-and-Deposits",
    )
    print(f"   resolved   : {off['resolved']}   reason: {off.get('reason')}")
    check("accounts page does NOT resolve a credit-cards task", off["resolved"] is False,
          "guards the category rubber-stamp fixed in PATCH 6")

    _head(4, "apply_validation + expand_plan")
    plan = apply_validation(plan, sub_goal, verdict)
    before = len(plan.sub_goals)
    plan = expand_plan(plan, verdict)
    print(f"   sub-goals  : {before} -> {len(plan.sub_goals)}")
    for sg in plan.sub_goals[:6]:
        print(f"     - [{sg.status.value:13}] {sg.question}")
    check("discovered entities became new sub-goals", len(plan.sub_goals) > before)

    _head(5, "synthesize")
    synthesis = synthesize(TASK, plan=plan, validated_results=[verdict])
    print(f"   claims     : {len(synthesis['claims'])}")
    print(f"   sources    : {synthesis['sources']}")
    print(f"   draft      : {synthesis['draft_answer'][:160]}")
    check("synthesis produced at least one claim", len(synthesis["claims"]) > 0)

    _head(6, "validate_answer + finalize (anti-hallucination)")
    visited = [CARD_LIST_URL]
    verification = validate_answer(synthesis["claims"], visited, strict=True)
    final = finalize(synthesis, verification)
    meta = final.get("metadata", {})
    print(f"   support    : {meta.get('claims_supported')}/{meta.get('claims_total')}"
          f"  (rate {meta.get('support_rate')})")
    print(f"   sources    : {final['source_urls']}")
    print(f"   not_found  : {final['not_found'][:2]}")
    check("finalize surfaces support_rate", "support_rate" in meta,
          "added in PATCH 14")
    check("every cited source was actually visited",
          all(u in visited for u in final["source_urls"]))

    _head(7, "a claim from an unvisited page is stripped, not cited")
    forged = list(synthesis["claims"]) + [{
        "id": "c_forged",
        "statement": "Banque Misr offers a 0% APR platinum card.",
        "source_url": "https://www.banquemisr.com/Home/Never/Visited",
    }]
    v2 = validate_answer(forged, visited, strict=True)
    f2 = finalize({"draft_answer": "Banque Misr offers a 0% APR platinum card.",
                   "sources": [], "missing_info": []}, v2)
    check("unvisited claim does not appear in the answer prose",
          "0% APR platinum" not in f2["answer"], repr(f2["answer"][:70]))
    check("unvisited URL is not cited",
          "Never/Visited" not in " ".join(f2["source_urls"]))

    print()
    if failures:
        print(f"FAILED ({len(failures)}): " + "; ".join(failures))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
