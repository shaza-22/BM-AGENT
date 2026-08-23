"""The frontend's text repairs, checked against the real file.

These run the actual JavaScript out of ``frontend/index.html`` under node, so
they test what ships rather than a copy of it. Skipped where node is absent.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import pytest

INDEX = pathlib.Path(__file__).resolve().parent.parent / "frontend" / "index.html"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def run_js(expression: str) -> str:
    """Evaluate an expression with the page's script in scope."""
    script = re.search(r"<script>(.*?)</script>", INDEX.read_text(encoding="utf-8"), re.S)
    assert script, "no script block in frontend/index.html"
    # The page's own body touches `document`, so take only the pure text
    # helpers: one contiguous run from `esc` to the first constant after them.
    body = script.group(1)
    start, end = body.index("const esc = "), body.index("const SG_MARK")
    source = body[start:end]
    assert "humaniseLabels" in source, "the helpers moved; update this slice"
    result = subprocess.run(
        ["node", "-e", f"{source}\nprocess.stdout.write(String({expression}))"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


class TestIdentifierRepairInSavedAnswers:
    """An answer recorded before the composer fix has the identifier baked in.

    A recording is replayed verbatim, so the repair also has to happen at
    render time or every saved run keeps showing it.
    """

    def test_a_leaked_identifier_is_rendered_as_words(self):
        out = run_js("humaniseLabels('entity_list — Credit Card, Debit cards.')")
        assert out == "Products listed on this page — Credit Card, Debit cards."

    def test_it_works_with_a_colon_and_with_a_bullet(self):
        assert "Products listed on this page:" in run_js("humaniseLabels('entity_list: A, B.')")
        assert "- Products listed on this page —" in run_js("humaniseLabels('- entity_list — A.')")

    @pytest.mark.parametrize("line", [
        "Issuance — EGP 250.",
        "Supplementary cards issuance and renewal — EGP100.",
        "Interest rate — 4% monthly.",
        "Cash withdrawals through BM ATMs and P.O.S and other banks — 2% minimum EGP15.",
    ])
    def test_real_fact_lines_are_untouched(self, line: str):
        """The repair must not be able to damage a genuine answer."""
        assert run_js(f"humaniseLabels({line!r})") == line

    def test_an_identifier_mid_sentence_is_left_alone(self):
        """Anchored to line start, so prose cannot be rewritten by accident."""
        line = "The card_type is described on the page — EGP 250."
        assert run_js(f"humaniseLabels({line!r})") == line

    def test_an_unknown_identifier_still_becomes_words(self):
        assert run_js("humaniseLabels('some_future_field: 12')") == "Some future field: 12"
