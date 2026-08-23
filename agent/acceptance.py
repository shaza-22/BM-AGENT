"""
The acceptance gate: a second opinion on ``resolved``, measured not guessed.

What it does
    Watches the pages a run fetches and, when the validator says a page
    resolves a sub-goal, checks whether the evidence it cited is content or
    chrome. Evidence that appears on page after page is boilerplate, whatever
    it says.

Inputs / Outputs
    ``observe(url, text)`` for every page fetched.
    ``judge(verdict, url, text)`` -> :class:`GateDecision`.

Why it is needed
    The vendored validator is deterministic and decides when navigation stops.
    Its bugs were patched (``src/person_b/PATCHES.md``), but the failure mode
    it exhibits -- resolving on text that happens to be present rather than on
    an answer -- is one this side can also test for, from a vantage point the
    validator does not have: it sees one page, this sees the run.

    That is the whole argument for keeping both. The validator asks *does this
    page answer the question*. The gate asks *is the thing you are pointing at
    even specific to this page*. Neither subsumes the other, and the gate
    contains no product knowledge at all -- it counts repetitions.

Why repetition is the signal
    Every page on this site carries the full global navigation and footer:
    profiling the fixtures found 60 of 173 distinct link keys present on all
    seven pages. Boilerplate is, by definition, the text that does not vary
    between pages. So a snippet that also appears on other pages fetched this
    run cannot be what makes *this* page the answer -- no vocabulary needed,
    and deleting any product word from the repo changes nothing here.

What it deliberately does not do
    It never promotes. The gate can turn ``resolved`` into "keep looking"; it
    can never turn anything into ``resolved``. One-way precedence is the same
    rule the navigator applies to the selector, for the same reason: exactly
    one component in the system is allowed to claim a sub-goal is answered.

Its blind spot, stated
    On the first page of a run there is nothing to compare against, so
    everything looks unique and the gate abstains. It gets sharper as a run
    goes on. That is the opposite of what you would want -- hop 0 is where a
    false resolve is most damaging -- which is exactly why the validator's own
    boilerplate fix (PATCH 5) had to land as well, rather than leaning on this.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from agent import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateDecision:
    """Whether a ``resolved`` verdict survives, and the count behind it."""

    accepted: bool
    reason: str
    snippets_checked: int = 0
    snippets_unique: int = 0
    pages_compared: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "snippets_checked": self.snippets_checked,
            "snippets_unique": self.snippets_unique,
            "pages_compared": self.pages_compared,
        }


def _normalise(text: str) -> str:
    """Collapse whitespace and case so the same chrome compares equal.

    Two renderings of one nav item differ by indentation and stray spacing far
    more often than by wording, and a snippet that fails to match its own twin
    would read as unique -- the gate's failure direction has to be *abstain*,
    not *false confidence*.
    """
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _evidence_snippets(verdict: dict[str, Any]) -> list[str]:
    """The text a verdict is standing on, from wherever it recorded it.

    Person B's evidence records carry ``evidence_text``; some carry only a
    ``value``. Both are accepted, since either is a claim about what is on the
    page. Anything too short to be distinctive is dropped rather than counted
    as boilerplate -- "Cash" appearing on every page proves nothing.
    """
    snippets: list[str] = []
    for item in verdict.get("evidence") or ():
        if not isinstance(item, dict):
            item = getattr(item, "to_dict", lambda: {})()
        for key in ("evidence_text", "value"):
            raw = item.get(key)
            if isinstance(raw, str):
                text = _normalise(raw)
                if len(text) >= config.GATE_MIN_SNIPPET_CHARS:
                    snippets.append(text)
                    break
    return snippets


@dataclass
class AcceptanceGate:
    """Per-run boilerplate detector. One instance per plan, not per sub-goal.

    It spans the whole plan on purpose: sub-goal two benefits from the pages
    sub-goal one fetched, and chrome is chrome across the run.
    """

    enabled: bool = True
    min_repeats: int = config.GATE_MIN_REPEATS
    _pages: dict[str, str] = field(default_factory=dict, repr=False)

    # -- observation -------------------------------------------------------
    def observe(self, url: str, text: str | None) -> None:
        """Record a fetched page. Cheap, and called for every page."""
        if not url or not text:
            return
        self._pages.setdefault(url, _normalise(text))

    @property
    def pages_seen(self) -> int:
        return len(self._pages)

    # -- judgement ---------------------------------------------------------
    def judge(self, verdict: dict[str, Any], url: str, text: str | None = None) -> GateDecision:
        """Decide whether a ``resolved`` verdict is standing on page content.

        Only ``resolved`` is examined. A verdict that already declines to
        resolve needs no help declining, and running the check anyway would
        invite the gate to start editing verdicts it was not asked about.
        """
        if not self.enabled:
            return GateDecision(True, "gate disabled")
        if not verdict.get("resolved"):
            return GateDecision(True, "not a resolve; nothing to gate")

        others = [body for other_url, body in self._pages.items() if other_url != url]
        if len(others) < self.min_repeats:
            # Abstaining is the honest answer, and it is stated rather than
            # dressed up as a pass: with too few pages to compare, repetition
            # is not yet measurable.
            return GateDecision(
                True,
                f"only {len(others)} other page(s) seen; too few to tell content from chrome",
                pages_compared=len(others),
            )

        snippets = _evidence_snippets(verdict)
        if not snippets:
            # No quotable evidence to test. Deciding on absence would make the
            # gate an opinion about verdict shape rather than about content.
            return GateDecision(
                True,
                "verdict cites no quotable evidence; nothing to measure",
                pages_compared=len(others),
            )

        unique = [s for s in snippets if sum(s in body for body in others) < self.min_repeats]
        decision = GateDecision(
            accepted=bool(unique),
            reason=(
                f"{len(unique)}/{len(snippets)} evidence snippets are specific to this page"
                if unique
                else (
                    f"all {len(snippets)} evidence snippets also appear on "
                    f"{self.min_repeats}+ other pages fetched this run, so they are "
                    "site chrome rather than an answer"
                )
            ),
            snippets_checked=len(snippets),
            snippets_unique=len(unique),
            pages_compared=len(others),
        )
        return decision


def gate_from_config(*, enabled: bool | None = None) -> AcceptanceGate:
    """Build the gate as configured. Ships enabled."""
    return AcceptanceGate(
        enabled=config.ACCEPTANCE_GATE_ENABLED if enabled is None else enabled,
        min_repeats=config.GATE_MIN_REPEATS,
    )
