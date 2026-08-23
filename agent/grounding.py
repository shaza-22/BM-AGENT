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
# Sentence boundaries. A terminator only ends a sentence when whitespace or the
# end of the line follows it -- otherwise "2.81%" splits into "2." and "81%",
# and the fragments then fail the figure check for numbers the model never
# wrote. Same reason "P.O.S" must survive intact.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Sentences, with each line its own unit.

    Lines are kept separate so a struck bullet or heading does not take its
    neighbours with it, and so markdown structure survives the round trip.
    """
    out: list[str] = []
    for line in (text or "").splitlines():
        if not line.strip():
            out.append("")          # preserve blank lines for paragraphing
            continue
        out.extend(part for part in _SENTENCE_BREAK.split(line.strip()) if part.strip())
    return out

MIN_ANCHOR_CHARS = 4

# A label made only of digits ("3", from a Tenor column) is not a label a
# sentence can be said to "name" -- every figure in the answer matches it, and
# the pair rule then strikes correct sentences for not quoting its value.
_NUMERIC_LABEL = re.compile(r"^[\d\s.,%-]+$")

# --- Scaffolding -----------------------------------------------------------
# The rule "every sentence must carry a grounded label or value" is right for
# assertions and wrong for the sentences that hold an answer together. It
# struck every opening line, grouping line and closing note, because none of
# them contains a figure -- so what survived was the bare fact sentences, and
# the answer read as a list. Measured on a live run: 7 of 20 sentences removed,
# every one of them for "carries no label or value".
#
# The distinction is not "does this contain a digit" but "does this assert
# something about the bank". So scaffolding is recognised *positively*, by
# structure, and everything else keeps the old rule. An allow-list fails safe:
# a sentence that is not recognisably scaffolding is still an assertion and
# still has to be grounded.
#
# "There is no annual fee." is the case that must keep failing. It asserts a
# fact, carries no figure, and is exactly what a model reaches for when the
# evidence does not cover something -- so negation disqualifies a sentence from
# scaffolding regardless of anything else.

# Words that turn a sentence into a claim about what is or is not the case.
_NEGATION = re.compile(
    r"\b(no|not|none|never|without|free|excluded|included|unlimited|any|"
    r"isn'?t|aren'?t|doesn'?t|don'?t|won'?t|cannot|can'?t)\b"
)

# Talking about the answer or its sources rather than about the bank.
_META = re.compile(
    r"\b(here (are|is)|below|above|as follows|the following|these (figures|are|fees|rates)|"
    r"this (answer|information|is based)|according to|taken from|sourced from|"
    r"from the (page|source|card'?s|bank'?s)|listed|shown|see the source|"
    r"summar(y|ised|ized)|breakdown|in summary|note that|all figures|"
    r"the figures (above|below)|source[sd]?)\b"
)

# A heading or lead-in: ends with a colon, or is a short markdown heading or
# bullet header. These introduce facts, they do not assert them.
_LEAD_IN = re.compile(r":\s*$")
_MARKDOWN_HEADING = re.compile(r"^\s*(#{1,6}\s+|\*\*[^*]+\*\*\s*:?\s*$|[-*]\s*\*\*)")


def is_scaffolding(sentence: str) -> bool:
    """Does this sentence hold the answer together without asserting a fact?

    True only for structure: a lead-in ending in a colon, a markdown heading,
    or a sentence that talks about the answer and its sources. Never true for
    a sentence containing a figure, and never true for one containing a
    negation or an absolute -- those assert something and must be grounded.
    """
    stripped = sentence.strip()
    if not stripped:
        return True
    if _NUMBER.search(stripped):
        return False          # any figure makes it an assertion, checked above
    low = _norm(stripped)
    if _NEGATION.search(low):
        return False          # "there is no annual fee" is a claim, not framing
    if _MARKDOWN_HEADING.match(stripped) or _LEAD_IN.search(stripped):
        return True
    return bool(_META.search(low))


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
        # The label counts as evidence too: a Tenor row is labelled "3", so a
        # sentence quoting "3 — 2.81%" is quoting the row, not inventing a
        # figure. Scanning only values made every such row unquotable.
        for source in (value, label, str(claim.get("statement") or "")):
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

    for raw in _split_sentences(text):
        sentence = raw.strip()
        if not sentence:
            kept.append("")          # a blank line is layout, not a claim
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
            # No grounded token. Either it is structure holding the answer
            # together, which is allowed, or it is an unsupported assertion,
            # which is not. See is_scaffolding for where that line is drawn.
            if is_scaffolding(sentence):
                kept.append(sentence)
                continue
            struck.append((sentence, "asserts something the evidence does not support"))
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
            present = [
                (label, value) for label, value in pairs
                if label and label in low and not _NUMERIC_LABEL.match(label)
            ]
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

    # Rejoin on newlines so headings and grouped lists keep their shape; the
    # answer is rendered as markdown, and " ".join would flatten it into the
    # run-on paragraph this whole step exists to avoid.
    rendered = "\n".join(kept)
    rendered = re.sub(r"\n{3,}", "\n\n", rendered).strip()
    report = GroundingReport(text=rendered, kept=[k for k in kept if k], struck=struck)
    if struck:
        logger.info(
            "grounding struck %d/%d generated sentence(s): %s",
            len(struck), report.sentences_total, [reason for _s, reason in struck],
        )
    return report
