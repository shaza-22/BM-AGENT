"""
Link selection: deciding which link moves the agent closer to a sub-goal.

What it does
    Reduces the links found on a page to a shortlist, asks a language model to
    pick one, and returns that choice together with the model's stated reason.

Inputs
    ``select_next_link(sub_goal, candidates, visited, context, llm=...)``
    -- the sub-goal text, the frontier of :class:`Candidate` links, the set of
    canonical keys already visited, and a :class:`SelectionContext` describing
    where the run currently stands.

Outputs
    A :class:`Selection`: the chosen URL and label, the reasoning, a confidence,
    and how many candidates were offered versus available. ``url=None`` is a
    first-class result meaning "none of these lead anywhere useful" -- not a
    failure, and not a low-confidence guess.

Why it is needed
    This is the agentic step. Everything else in the project fetches, parses or
    validates; this is where the agent actually decides what to do next, and
    the ``reasoning`` it returns is the record of why.

Topic-agnosticism
    The ranking heuristic never reads ``sub_goal``. It scores links only on
    structure -- which page they were found on, how long ago, which region of
    the page, how often they have been offered. A lexical filter matching
    sub-goal words against labels would quietly become the agent, and would
    fail on any sub-goal phrased in different words from the site's own. The
    model does the semantic work; the ranking only decides what fits in the
    prompt. ``test_ranking_is_independent_of_the_sub_goal`` enforces this.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Sequence

from agent import config
from agent.llm import LLMClient
from browsing.extract_links import LinkDict, alias_key, canonical_key

logger = logging.getLogger(__name__)

# "choice" is an index into the offered list; -1 means none of them fit. An
# index cannot name a page that was never offered, which a URL string can --
# see the note on parse_selection.
SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": "integer"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["choice", "reasoning", "confidence"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are the navigation component of a research agent working \
through a bank's public website. You are given one sub-goal and a numbered list \
of links found on the pages visited so far. Choose the single link most likely \
to lead to a page that answers the sub-goal.

Rules:
- Answer with the NUMBER of a link from the list, or -1 if none of them plausibly \
lead toward the sub-goal.
- Prefer a link whose destination would contain the answer over one that merely \
mentions the topic.
- Category and list pages are useful stepping stones when no link answers the \
sub-goal directly.
- Choosing -1 is correct and expected when the sub-goal concerns something this \
website does not cover. Do not guess in that case.
- Explain your choice in one or two sentences. The explanation is shown to a \
human reviewing the agent's decisions, so state what you expect to find.

Reply with JSON only: {"choice": <int>, "reasoning": "<why>", "confidence": <0-1>}"""


@dataclass
class Candidate:
    """One link the agent could follow, with where and when it was discovered."""

    link: LinkDict
    discovered_on: str
    discovered_at_hop: int
    times_offered: int = 0

    @property
    def key(self) -> str:
        return self.link["key"]

    @property
    def url(self) -> str:
        return self.link["url"]

    @property
    def label(self) -> str:
        return self.link["label"]

    @property
    def source(self) -> str:
        return self.link["source"]


@dataclass
class SelectionContext:
    """Where the run stands, for scoring and for the prompt."""

    hop: int = 0
    pages_fetched: int = 0
    max_hops: int = config.MAX_HOPS
    max_pages: int = config.MAX_PAGES
    current_url: str | None = None
    trail: Sequence[str] = ()
    alias_visited: frozenset[str] = frozenset()
    limit: int = config.CANDIDATE_LIMIT


@dataclass
class Selection:
    """The outcome of one selection step."""

    url: str | None
    label: str | None
    reasoning: str
    confidence: float
    candidate_index: int | None = None
    offered: int = 0
    available: int = 0
    parse_error: str | None = None
    raw_response: str | None = None
    ranked: list[Candidate] = field(default_factory=list, repr=False)


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------
def score_candidate(candidate: Candidate, context: SelectionContext) -> float:
    """Structural score for one candidate. Never reads the sub-goal."""
    score = 0.0

    on_current_page = (
        context.current_url is not None
        and canonical_key(candidate.discovered_on) == canonical_key(context.current_url)
    )
    if on_current_page:
        # Keeps the default behaviour depth-first: links on the page we are
        # standing on outrank anything carried over from earlier pages.
        score += config.WEIGHT_CURRENT_PAGE

    age = max(0, context.hop - candidate.discovered_at_hop)
    score += config.WEIGHT_RECENCY / (1 + age)

    score += config.WEIGHT_SOURCE.get(candidate.source, 0.0)
    if candidate.source == "nav" and context.hop > 0:
        score -= config.PENALTY_NAV_AFTER_FIRST_HOP

    score -= config.PENALTY_PER_OFFER * candidate.times_offered

    alias = alias_key(candidate.url)
    if alias and alias in context.alias_visited:
        score -= config.PENALTY_ALIAS_VISITED

    return score


def rank_candidates(
    candidates: Sequence[Candidate], visited: frozenset[str] | set[str], context: SelectionContext
) -> list[Candidate]:
    """Drop visited links, score the rest, return the best ``context.limit``.

    Ties keep discovery order, which is document order within a page -- a weak
    but real relevance signal.
    """
    fresh = [candidate for candidate in candidates if candidate.key not in visited]
    ordered = [
        candidate
        for _, candidate in sorted(
            enumerate(fresh), key=lambda pair: (-score_candidate(pair[1], context), pair[0])
        )
    ]
    return _apply_region_quotas(ordered, context.limit)


def _apply_region_quotas(ordered: Sequence[Candidate], limit: int) -> list[Candidate]:
    """Trim to ``limit`` while keeping every page region represented.

    A body-heavy page crowds the others out completely: on the real homepage a
    straight top-N cut offered 40 body links and nothing else, so the
    footer-only fees hub could not be chosen at all on the first hop.

    Reservations are filled round-robin across regions and may claim at most
    half the slots, so a large reserve can never starve the score ranking
    itself -- the remaining slots always go to the best-scoring candidates
    regardless of region. The result stays in score order, so a reserved link
    sits at its natural rank rather than being promoted to the top.
    """
    if len(ordered) <= limit:
        return list(ordered)

    pools = {
        region: [candidate for candidate in ordered if candidate.source == region][:reserve]
        for region, reserve in config.REGION_RESERVED_SLOTS.items()
    }
    reserve_budget = max(1, limit // 2)

    chosen: set[int] = set()
    while len(chosen) < reserve_budget and any(pools.values()):
        for pool in pools.values():
            if not pool or len(chosen) >= reserve_budget:
                continue
            chosen.add(id(pool.pop(0)))

    for candidate in ordered:
        if len(chosen) >= limit:
            break
        chosen.add(id(candidate))

    return [candidate for candidate in ordered if id(candidate) in chosen][:limit]


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------
def _path_of(url: str) -> str:
    """Show the path only; the host is constant and would cost ~25 chars a line."""
    return re.sub(r"^https?://[^/]+", "", url) or "/"


def build_prompt(sub_goal: str, ranked: Sequence[Candidate], context: SelectionContext) -> str:
    lines = [f"Sub-goal: {sub_goal}", ""]

    if context.trail:
        lines.append("Pages visited so far this run:")
        lines.extend(f"  {step}" for step in context.trail)
        lines.append("")

    lines.append(
        f"Hop {context.hop + 1} of {context.max_hops}; "
        f"{context.pages_fetched} of {context.max_pages} pages used."
    )
    if context.current_url:
        lines.append(f"Currently on: {_path_of(context.current_url)}")
    lines.append("")
    lines.append("Links available (number | label | page region | path):")

    for index, candidate in enumerate(ranked):
        marker = " [PDF]" if candidate.link["is_pdf"] else ""
        origin = (
            ""
            if context.current_url
            and canonical_key(candidate.discovered_on) == canonical_key(context.current_url)
            else f" (seen at hop {candidate.discovered_at_hop + 1})"
        )
        lines.append(
            f"{index} | {candidate.label} | {candidate.source} | "
            f"{_path_of(candidate.url)}{marker}{origin}"
        )

    lines.append("")
    lines.append("Which link should the agent follow next?")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _extract_json(raw: str) -> dict:
    text = _FENCE_RE.sub("", raw.strip())
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Some models wrap JSON in prose. Take the outermost braced span.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("expected a JSON object", text, 0)
    return parsed


def _coerce_index(value: object) -> int:
    if isinstance(value, bool):  # bool is an int subclass; not a valid index
        raise ValueError("choice must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        return int(value.strip())
    raise ValueError(f"choice is not an integer: {value!r}")


def _coerce_confidence(value: object) -> float:
    try:
        confidence = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.5
    return min(1.0, max(0.0, confidence))


def parse_selection(raw: str, ranked: Sequence[Candidate], available: int) -> Selection:
    """Turn a model reply into a :class:`Selection`, never raising.

    The reply names a candidate by index rather than by URL. A model asked for
    a URL will occasionally invent one, or reword an encoded path into
    something that no longer resolves; an index either points at a link the
    agent actually found or is out of range, which is detectable. An
    unparseable reply becomes "no candidate" with the error in ``reasoning``,
    so a bad response costs one hop instead of the task.
    """
    base = {
        "offered": len(ranked),
        "available": available,
        "raw_response": raw,
        "ranked": list(ranked),
    }

    try:
        parsed = _extract_json(raw)
        index = _coerce_index(parsed.get("choice"))
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("could not parse model reply (%s): %r", exc, raw[:200])
        return Selection(
            url=None,
            label=None,
            reasoning=f"Could not parse the model's reply ({exc}); treating as no candidate.",
            confidence=0.0,
            parse_error=str(exc),
            **base,
        )

    reasoning = str(parsed.get("reasoning") or "").strip()
    confidence = _coerce_confidence(parsed.get("confidence"))

    if index == -1:
        return Selection(
            url=None,
            label=None,
            # The reasoning field is a project deliverable and is never allowed
            # to be empty, whatever the model returned.
            reasoning=reasoning or "The model reported that no listed link leads toward the sub-goal.",
            confidence=confidence,
            candidate_index=-1,
            **base,
        )

    if not 0 <= index < len(ranked):
        logger.warning("model chose out-of-range index %s of %d", index, len(ranked))
        return Selection(
            url=None,
            label=None,
            reasoning=(
                f"The model chose link {index}, which was not on the list of "
                f"{len(ranked)} offered; treating as no candidate."
            ),
            confidence=0.0,
            parse_error=f"index {index} out of range",
            **base,
        )

    chosen = ranked[index]
    return Selection(
        url=chosen.url,
        label=chosen.label,
        reasoning=reasoning or f"Selected {chosen.label!r} but gave no explanation.",
        confidence=confidence,
        candidate_index=index,
        **base,
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def select_next_link(
    sub_goal: str,
    candidates: Sequence[Candidate],
    visited: frozenset[str] | set[str],
    context: SelectionContext,
    *,
    llm: LLMClient,
) -> Selection:
    """Pick the next hop, or report that no candidate fits."""
    available = sum(1 for candidate in candidates if candidate.key not in visited)
    ranked = rank_candidates(candidates, visited, context)

    if not ranked:
        return Selection(
            url=None,
            label=None,
            reasoning="No unvisited links remain to choose from.",
            confidence=1.0,
            offered=0,
            available=available,
        )

    if len(ranked) < available:
        logger.info(
            "trimmed candidates for hop %d: offered %d of %d available",
            context.hop + 1, len(ranked), available,
        )

    prompt = build_prompt(sub_goal, ranked, context)
    raw = llm.complete(prompt, system=SYSTEM_PROMPT, schema=SELECTION_SCHEMA)
    selection = parse_selection(raw, ranked, available)

    if selection.url and selection.confidence < config.MIN_CONFIDENCE:
        logger.info(
            "discarding pick %r: confidence %.2f below %.2f",
            selection.label, selection.confidence, config.MIN_CONFIDENCE,
        )
        return Selection(
            url=None,
            label=None,
            reasoning=(
                f"Chose {selection.label!r} with confidence {selection.confidence:.2f}, "
                f"below the configured minimum of {config.MIN_CONFIDENCE:.2f}."
            ),
            confidence=selection.confidence,
            candidate_index=selection.candidate_index,
            offered=selection.offered,
            available=selection.available,
            raw_response=raw,
            ranked=selection.ranked,
        )

    return selection
