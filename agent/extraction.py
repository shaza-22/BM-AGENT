"""
Reading a page with the model when keyword extraction found nothing.

What it does
    Given a page's text and the sub-goal it was fetched for, asks the model for
    ``{label, value}`` facts, then throws away every fact whose value is not on
    the page **verbatim**.

Inputs / Outputs
    ``extract_facts(sub_goal, page_text, llm, ...)`` -> :class:`Extraction`.

Why it is needed
    The vendored extraction is keyword matching, and every visible failure of
    this system falls out of that one property: an Arabic page yields nothing
    because there are no Arabic keywords; a fees page reads as not-found
    because it says "Fees and Rates" and the question said "fees and charges";
    a sub-goal resolves on entity discovery and then reports no verified facts
    because no field was extracted.

    Measured on the Arabic accounts page, deterministic extraction returns
    ``entities: 0, tables: 0, sections: 0, fields: 0`` -- while the page text
    itself decodes perfectly. The text was never the problem. Reading it was.

    A model reading the page needs no vocabulary table, so this is one general
    fix rather than four special cases: language-agnostic, paraphrase-tolerant,
    and able to read prose rather than only tables.

What makes it safe: the label may be written, the value may not
    The label is the model's -- that is exactly how it bridges "Fees and Rates"
    on the page to "fees and charges" in the question. The **value** must be a
    literal substring of the page after whitespace normalisation. Not
    paraphrased, not reformatted, not rounded.

    A model asked to extract will sometimes helpfully infer, and an inferred
    figure with a citation attached is the worst output this system could
    produce. Anything failing the check is **discarded, never repaired** --
    repairing it would be this module deciding what the page meant.

Cost, and when it fires
    One call per sub-goal, at most, and only after the deterministic path has
    actually failed. It runs once when a sub-goal's navigation ends, on the
    best page that navigation saw -- not inside ``validate_fn``, which the
    navigator calls for every page it fetches and which would have made this
    one call per page.

Failing safe
    Every failure -- unreachable model, unparseable reply, no facts, every
    value struck -- leaves the run exactly as it was. This can only add facts.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from agent import config
from agent.llm import LLMClient, LLMError

logger = logging.getLogger(__name__)

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["label", "value"],
                "additionalProperties": False,
            },
        },
        "present": {"type": "boolean"},
        "note": {"type": "string"},
    },
    "required": ["facts", "present", "note"],
    "additionalProperties": False,
}

SYSTEM = (
    "You read one page of a bank's website and pull out the facts that answer a "
    "specific question. You copy values exactly as the page writes them. You "
    "never calculate, convert, round, reformat or infer a value, and when the "
    "page does not answer the question you say so instead of finding something "
    "close."
)

PROMPT = """\
Question to answer from this page:
{question}

Page text:
---
{page_text}
---

Pull out the facts on this page that answer the question, as label and value pairs.

Rules:
- The **value** must be copied from the page character for character. If the page
  says "EGP 250", write "EGP 250" — not "250 EGP", not "EGP250", not "250".
- The **label** is yours to write. Name the thing the way the question would,
  even if the page words it differently.
- Do not calculate, total, convert or round anything. Copy only.
- Include only what bears on the question. A page has many numbers on it.
- If the page does not answer the question, set present to false and return no
  facts. That is a useful answer and it is often the correct one.
"""

# Characters that differ between a page's rendering and a model's copy of it
# without changing what the value says. Normalising both sides for the match
# keeps the check strict about content and forgiving about typography.
_EQUIVALENT = {
    " ": " ", " ": " ", " ": " ",     # non-breaking spaces
    "‐": "-", "‑": "-", "‒": "-",     # dashes
    "–": "-", "—": "-", "−": "-",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "٫": ".", "٬": ",",                   # Arabic decimal separators
}
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

# Unrendered template markup. The site ships Vue that never rendered --
# "{{currencyCalculator.CashBuying}}" -- and it *is* in the page text, so the
# verbatim check confirms it is present and a citation gets attached to a
# variable name. Presence is not content, and this is the one shape where a
# value can be literally on the page and still not be a fact.
#
# Checked here, in synthesize.py for the deterministic path, and in
# grounding.py for anything that reaches prose. Three places on purpose: a
# check that exists on only one path is a check the other path bypasses.
PLACEHOLDER = re.compile(r"\{\{.*?\}\}|\{%.*?%\}|\$\{.*?\}|<%.*?%>|\[\[.*?\]\]")


def looks_like_placeholder(text: str) -> bool:
    return bool(PLACEHOLDER.search(text or ""))


def normalise(text: str) -> str:
    """Fold typography and whitespace, keep everything that carries meaning."""
    folded = "".join(_EQUIVALENT.get(ch, ch) for ch in (text or ""))
    return re.sub(r"\s+", " ", folded).strip().casefold()


@dataclass
class Fact:
    label: str
    value: str

    def to_dict(self) -> dict[str, str]:
        return {"label": self.label, "value": self.value}


@dataclass
class Extraction:
    """What the fallback produced, and what the verbatim check did to it."""

    facts: list[Fact] = field(default_factory=list)
    attempted: bool = False
    present: bool = True
    note: str = ""
    reason: str = ""
    offered: int = 0                     # facts the model returned
    struck: list[tuple[str, str]] = field(default_factory=list)  # (label, value)

    @property
    def used(self) -> bool:
        return bool(self.facts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "used": self.used,
            "present": self.present,
            "reason": self.reason,
            "note": self.note,
            "offered": self.offered,
            "kept": len(self.facts),
            "struck": [{"label": lab, "value": val} for lab, val in self.struck],
            "facts": [f.to_dict() for f in self.facts],
        }

    def as_table(self, table_name: str) -> dict[str, Any]:
        """The facts in the shape the existing claim pipeline already reads.

        A ``{label, value}`` table flows straight through PATCH 16's row
        reader, so the facts become claims with real values, get text-checked
        against the page by PATCH 17, and reach grounding and composition --
        with no new plumbing anywhere.
        """
        return {
            "table_name": table_name,
            "headers": ["label", "value"],
            "records": [{"label": f.label, "value": f.value} for f in self.facts],
        }


def _extract_json(raw: str) -> dict:
    text = _FENCE_RE.sub("", (raw or "").strip())
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("expected a JSON object", text, 0)
    return parsed


def occurs_verbatim(value: str, haystack: str) -> bool:
    """Is ``value`` on the page as a whole token run, not inside a larger one?

    A plain substring test is not enough. The page says "EGP 2500" and the
    model offers "250" -- "250" *is* a substring, so a substring check passes a
    figure that is off by a factor of ten. The match has to end where a token
    ends.

    Boundaries are only required on the sides where the value itself begins or
    ends with a word character, so "4%" and "(free)" still match.
    """
    needle = normalise(value)
    if not needle:
        return False
    pattern = re.escape(needle)
    if needle[0].isalnum():
        pattern = r"(?<!\w)" + pattern
    if needle[-1].isalnum():
        pattern = pattern + r"(?!\w)"
    return re.search(pattern, haystack) is not None


def verbatim_filter(facts: list[Fact], page_text: str) -> tuple[list[Fact], list[tuple[str, str]]]:
    """Keep only facts whose value is literally on the page. Returns (kept, struck)."""
    haystack = normalise(page_text)
    kept: list[Fact] = []
    struck: list[tuple[str, str]] = []
    for fact in facts:
        if looks_like_placeholder(fact.value) or looks_like_placeholder(fact.label):
            struck.append((fact.label, fact.value))
        elif occurs_verbatim(fact.value, haystack):
            kept.append(fact)
        else:
            struck.append((fact.label, fact.value))
    return kept, struck


def extract_facts(
    question: str,
    page_text: str,
    llm: LLMClient,
    *,
    max_page_chars: int = config.EXTRACTION_MAX_PAGE_CHARS,
    max_facts: int = config.EXTRACTION_MAX_FACTS,
) -> Extraction:
    """Read the page for facts answering ``question``, keeping only verbatim values."""
    result = Extraction()
    if not (page_text or "").strip():
        result.reason = "no page text to read"
        return result

    result.attempted = True
    prompt = PROMPT.format(question=question, page_text=page_text[:max_page_chars])

    try:
        raw = llm.complete(prompt, system=SYSTEM, schema=EXTRACTION_SCHEMA)
    except LLMError as exc:
        logger.warning("extraction fallback unavailable, keeping the current result: %s", exc)
        result.reason = f"model unavailable: {exc}"
        return result

    try:
        parsed = _extract_json(raw)
    except Exception as exc:
        logger.warning("extraction fallback returned unparseable output: %s", exc)
        result.reason = f"unparseable reply: {exc}"
        return result

    result.present = bool(parsed.get("present", True))
    result.note = str(parsed.get("note") or "").strip()

    offered: list[Fact] = []
    for item in parsed.get("facts") or ():
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        value = str(item.get("value") or "").strip()
        if label and value:
            offered.append(Fact(label=label, value=value))
        if len(offered) >= max_facts:
            break
    result.offered = len(offered)

    if not offered:
        result.reason = (
            "the model reports the page does not answer this" if not result.present
            else "the model returned no facts"
        )
        logger.info("extraction fallback: 0 facts offered — %s", result.reason)
        return result

    result.facts, result.struck = verbatim_filter(offered, page_text)

    # Logged at INFO and deliberately explicit about *which* of the two failure
    # modes happened, because they need opposite responses and look identical
    # from outside: "found nothing on the page" is the page's answer, while
    # "found things and the formatting was rejected" is this system's problem.
    if result.struck:
        logger.info(
            "extraction fallback: verbatim check REJECTED %d of %d value(s) — "
            "the model reformatted them rather than copying: %s",
            len(result.struck), result.offered,
            "; ".join(f"{lab!r}={val!r}" for lab, val in result.struck[:6]),
        )
    if result.facts:
        result.reason = f"kept {len(result.facts)} of {result.offered} fact(s)"
        logger.info(
            "extraction fallback: KEPT %d of %d fact(s) — %s",
            len(result.facts), result.offered,
            "; ".join(f"{f.label}={f.value}" for f in result.facts[:6]),
        )
    else:
        result.reason = (
            f"all {result.offered} value(s) failed the verbatim check; "
            "nothing added"
        )
        logger.warning(
            "extraction fallback: found %d fact(s) and kept NONE — every value "
            "failed the verbatim check. This is a formatting mismatch, not an "
            "empty page.", result.offered,
        )
    return result
