"""
Grounding: does this generated sentence only say what the evidence says?

What it does
    Takes prose written by a model and the claims it was supposed to be
    written from, and strikes every sentence that asserts something the claims
    do not support.

Inputs / Outputs
    ``check_grounding(text, claims)`` -> :class:`GroundingReport` with the
    surviving text, the struck sentences, and why each was struck.

Why it is needed
    The vendored verification step attributes claims to *source URLs*. That is
    enough when the claims are built mechanically from a page -- they cite the
    page they came from by construction -- but it is not a check on prose. A
    generated sentence can cite a page that was genuinely fetched and still say
    something the page never said, and URL attribution would pass it. Since the
    interface puts a percentage bar next to that number, the check has to be
    about the text.

What it checks
    Two things, both deterministic and both about *this* text against *these*
    claims. No vocabulary, no model, no network.

    1. **Every figure must be quoted.** Any number, amount or percentage in a
       sentence must appear in some claim's value. This is what stops a model
       inventing a fee, and it is the failure that matters most: a wrong number
       presented with a source is worse than no answer.
    2. **Every sentence must be anchored.** A sentence has to carry at least
       one label or value from the claim set. This is what stops a fluent
       bridging sentence -- "the card has no annual fee" -- from riding along
       carrying no checkable token at all.

What it does not catch, stated plainly
    Recombination. "Issuance: EGP 250" and "Renewal: EGP 250" are both real, so
    "the annual fee is EGP 250" uses only quoted figures and is still false.
    Requiring a label and its own value to co-occur narrows this -- and is
    checked below -- but a model that pairs a real label with a different real
    value from the same page will pass. Treat this as raising the cost of a
    hallucination, not as proof of correctness.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# A figure: 250, 10,000, 2.81%, 400000. Currency words travel with them in the
# text and are matched as part of the surrounding value, not separately.
_NUMBER = re.compile(r"\d[\d,.]*%?")
# Sentence boundaries, keeping list items and newlines as their own units so a
# struck bullet does not take its neighbours with it.
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?|\n")

MIN_ANCHOR_CHARS = 4


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().casefold()


def _norm_number(token: str) -> str:
    """Compare figures without their thousands separators or trailing dot."""
    return token.replace(",", "").rstrip(".").casefold()


@dataclass
class GroundingReport:
    """What survived, what did not, and why."""

    text: str
    kept: list[str] = field(default_factory=list)
    struck: list[tuple[str, str]] = field(default_factory=list)  # (sentence, reason)

    @property
    def sentences_total(self) -> int:
        return len(self.kept) + len(self.struck)

    @property
    def rate(self) -> float:
        return len(self.kept) / self.sentences_total if self.sentences_total else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentences_total": self.sentences_total,
            "sentences_kept": len(self.kept),
            "sentences_struck": len(self.struck),
            "rate": round(self.rate, 4),
            "struck": [{"sentence": s, "reason": r} for s, r in self.struck],
        }


def _claim_parts(claims: Iterable[dict]) -> tuple[set[str], list[str], list[tuple[str, str]]]:
    """(quotable figures, anchors, label/value pairs) from the claim set."""
    figures: set[str] = set()
    anchors: list[str] = []
    pairs: list[tuple[str, str]] = []
    for claim in claims:
        value = str(claim.get("value") or "")
        label = str(claim.get("entity") or claim.get("field") or "")
        for source in (value, str(claim.get("statement") or "")):
            for match in _NUMBER.findall(source):
                figures.add(_norm_number(match))
        for candidate in (value, label):
            normalised = _norm(candidate)
            if len(normalised) >= MIN_ANCHOR_CHARS:
                anchors.append(normalised)
        if value and label:
            pairs.append((_norm(label), _norm(value)))
    return figures, anchors, pairs


def check_grounding(text: str, claims: list[dict]) -> GroundingReport:
    """Strike every sentence the claims do not support."""
    if not text or not text.strip():
        return GroundingReport(text="")
    if not claims:
        # Nothing to ground against. Returning the text unchanged here would
        # make this function a rubber stamp in exactly the case where the model
        # has the least to go on, so it strikes everything instead. The caller
        # is expected never to generate with no claims (see agent/answer.py).
        return GroundingReport(text="", struck=[(text.strip(), "no claims to ground against")])

    figures, anchors, pairs = _claim_parts(claims)
    kept: list[str] = []
    struck: list[tuple[str, str]] = []

    for raw in _SENTENCE.findall(text):
        sentence = raw.strip()
        if not sentence:
            continue
        low = _norm(sentence)

        unquoted = [
            token for token in _NUMBER.findall(sentence)
            if _norm_number(token) not in figures
        ]
        if unquoted:
            struck.append((sentence, f"figure(s) not in the evidence: {', '.join(unquoted)}"))
            continue

        if not any(anchor in low for anchor in anchors):
            struck.append((sentence, "carries no label or value from the evidence"))
            continue

        # If the sentence names a label, the value it puts next to that label
        # must be that label's own value. Catches the cheapest recombination:
        # a real fee name paired with a different real amount.
        #
        # Only *maximal* matching labels are checked. Labels nest -- "Issuance"
        # is a substring of "Supplementary cards issuance and renewal" -- so
        # checking every match would strike a correct sentence for failing to
        # quote the shorter label's value. The longest label present is the one
        # the sentence is actually about.
        if any(_norm_number(token) in figures for token in _NUMBER.findall(sentence)):
            present = [(label, value) for label, value in pairs if label and label in low]
            maximal = [
                (label, value) for label, value in present
                if not any(label != other and label in other for other, _v in present)
            ]
            mismatched = [
                label for label, value in maximal
                if not any(v in low for l, v in pairs if l == label)
            ]
            if mismatched:
                struck.append(
                    (sentence, f"pairs {mismatched[0]!r} with a figure that is not its value")
                )
                continue

        kept.append(sentence)

    report = GroundingReport(text=" ".join(kept).strip(), kept=kept, struck=struck)
    if struck:
        logger.info(
            "grounding struck %d/%d generated sentence(s): %s",
            len(struck), report.sentences_total, [reason for _s, reason in struck],
        )
    return report
