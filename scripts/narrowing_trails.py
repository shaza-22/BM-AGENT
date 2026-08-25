"""
Before/after navigation trails for narrowing follow-up questions.

What it does
    Runs one question twice against the saved fixtures -- once with
    ``DEEPEN_ON_NARROWING`` off, once on -- and prints both trails side by
    side, with what the validator extracted at the page each run stopped on.

Inputs
    ``fixtures/live/`` (see scripts/save_fixtures.py) and the questions in
    ``CASES`` below. Pass questions on the command line to use your own.

Outputs
    A trail per run on stdout, and the extracted entity names, so "it went one
    hop deeper" can be checked against "it found something more specific".

Why it is needed
    The deepening rule is the kind of change that is easy to claim and hard to
    show. This makes the claim falsifiable in one command, with no model calls
    and no quota spent: the deeper hop is chosen from the URL structure, so
    navigation here is fully deterministic.

Usage
    python scripts/save_fixtures.py        # once, to snapshot the pages
    python scripts/narrowing_trails.py
    python scripts/narrowing_trails.py "what about debit card types"

Note
    A hop that prints ``[FAIL 404]`` means the deeper page is not in
    fixtures/live/, not that the site lacks it -- the run then falls back to
    the verdict it was holding, which is the safety property, not the feature.
    Re-run save_fixtures.py to snapshot the paths listed in
    fixtures/fixture_urls.txt.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agent import config  # noqa: E402
from agent.llm import FakeLLMClient  # noqa: E402
from agent.navigator import Navigator  # noqa: E402
from person_b.api import plan_task, validate as person_b_validate  # noqa: E402
from tests.conftest import live_fetcher  # noqa: E402

SITE = "https://www.banquemisr.com"

# Each case seeds at the hub the question narrows from, because that is the
# page the bug was reported on. Reaching the hub is ordinary navigation and is
# not what this script is testing.
CASES: list[tuple[str, str]] = [
    ("what about debit card types", "/Home/SMEs/Retail%20Banking/Pages/Cards"),
    ("what savings accounts do you have", "/Home/SMEs/Retail%20Banking/Accounts%20And%20Deposits"),
    ("what personal loans are available", "/Home/SMEs/Retail%20Banking/Consumer%20Loans"),
    ("what credit cards do you have", "/Home/SMEs/Retail%20Banking/Pages/Cards"),
    # The control. A question no narrower than the page it stands on must come
    # out of this byte-identical, or the rule is over-navigating.
    ("what card types do you have", "/Home/SMEs/Retail%20Banking/Pages/Cards"),
]


def validator_for(sub_goal):
    def validate(_text, page):
        return person_b_validate(
            sub_goal, page.get("text") or "",
            source_url=page.get("url") or "",
            content_type="pdf" if "pdf" in (page.get("content_type") or "") else "text",
            status_code=page.get("status"),
        )
    return validate


def entity_names(result) -> list[str]:
    entities = (result.extracted or {}).get("entities") or []
    return [e.get("name") if isinstance(e, dict) else e for e in entities]


def run(question: str, seed_path: str) -> None:
    sub_goal = plan_task(question).sub_goals[0]
    print(f"\n=== {question!r}")
    print(f"    sub-goal      : {sub_goal.question}")
    print(f"    target_fields : {sub_goal.target_fields}")
    print(f"    seeded at     : {seed_path}")

    for deepen in (False, True):
        config.DEEPEN_ON_NARROWING = deepen
        # No scripted replies: a model call would raise. The deeper hop is read
        # off the URL structure, never asked for.
        result = Navigator(
            FakeLLMClient([]), fetcher=live_fetcher(),
            validate_fn=validator_for(sub_goal), seed_url=SITE + seed_path,
        ).navigate(sub_goal.question)

        label = "AFTER  (deepen on) " if deepen else "BEFORE (deepen off)"
        print(f"    {label}: {result.status}, "
              f"{result.pages_fetched} page(s), {result.hops_used} hop(s)")
        for step in result.trail:
            state = "ok" if step.fetch_ok else f"FAIL {step.fetch_status}"
            via = f"  <- {step.label}" if step.label else ""
            print(f"        hop {step.hop}: {step.url[len(SITE):] or '/'}  [{state}]{via}")
        names = entity_names(result)
        print(f"        stopped on : {(result.page or {}).get('url', '-')[len(SITE):] or '-'}")
        print(f"        reason     : {result.final_reasoning[:96]}")
        print(f"        found      : {len(names)} entities: {names}")
    config.DEEPEN_ON_NARROWING = True


def main() -> int:
    if not (ROOT / "fixtures" / "live" / "manifest.json").is_file():
        print("fixtures/live/ is empty -- run scripts/save_fixtures.py first.")
        return 1
    cases = [(q, dict(CASES).get(q, CASES[0][1])) for q in sys.argv[1:]] or CASES
    for question, seed in cases:
        run(question, seed)
    print("\nA [FAIL 404] hop means that page is not in fixtures/live/, not that")
    print("the site lacks it. The run then falls back to the verdict it held.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
