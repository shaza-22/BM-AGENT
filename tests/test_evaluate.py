"""The evaluation harness, run as part of the suite so it cannot rot.

``scripts/evaluate.py`` is the assignment's Evaluation Pipeline: one case per
requirement category, run end to end against the saved pages. A harness nobody
runs is a harness that stops working, so every case runs here too.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import evaluate  # noqa: E402


@pytest.mark.parametrize("case", evaluate.CASES, ids=lambda c: c.name)
def test_each_evaluation_case_passes(case) -> None:
    ok, detail, _result = evaluate.run_case(case)
    assert ok, f"{case.category}/{case.name}: {detail}"


def test_every_requirement_category_is_covered() -> None:
    """The point of the harness: breadth, not one task run five ways."""
    categories = {c.category for c in evaluate.CASES}
    assert {"lookup", "comparison", "missing-information trap", "context"} <= categories


def test_the_trap_case_guards_the_invariant() -> None:
    """A correct "not found" must never become a guess.

    This is the behaviour every other fix has to preserve, so it is asserted
    separately from the harness run: no claims, no generated prose.
    """
    case = next(c for c in evaluate.CASES if c.category == "missing-information trap")
    _ok, _detail, result = evaluate.run_case(case)
    assert result.resolved_count == 0
    assert result.claims_total == 0
    assert result.answer_source == "template"
