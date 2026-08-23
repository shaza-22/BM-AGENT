"""
Conversation memory: turning a follow-up into a standalone sub-goal.

What it does
    Holds the turns of one conversation and rewrites a follow-up ("what about
    the second one?", "and the fees for that?") into a sub-goal the navigator
    can act on without any of the surrounding conversation.

Inputs
    ``resolve_task(task, session, llm)`` -- the raw text the user typed, the
    session so far, and any :class:`~agent.llm.LLMClient`.
    ``Session.add_turn(task, resolution, result)`` -- a finished navigation.

Outputs
    ``resolve_task`` -> a :class:`Resolution`: the standalone ``sub_goal``,
    whether prior context was actually used, and the model's reasoning.
    ``Session.turns`` -> the conversation, each turn holding a
    :class:`TurnSummary` rather than a full navigation result.

Why it is needed
    The assignment requires follow-up instructions to work. Without this, every
    request is independent and "what about the second one?" navigates for the
    literal phrase, which resolves to nothing.

Why an LLM call rather than rules
    Pronoun and ellipsis resolution by pattern matching would be brittle, and
    the patterns would inevitably encode the site's vocabulary -- exactly the
    topic coupling the rest of this project avoids. The resolved sub-goal is
    always surfaced to the user, which is the real guard: a wrong rewrite is
    visible rather than silent.

Memory policy
    Session-scoped only. Nothing is written to disk and nothing survives the
    process. A turn stores a summary -- statuses, source URLs, and each hop's
    label and reasoning -- never a ``PageDict``: page text reaches 140k
    characters and raw HTML 300KB, and keeping those per turn would both leak
    memory and dump hundreds of KB into anything that serialises a turn.

On carrying prior-visited URLs into a follow-up
    A follow-up about a page found in an earlier turn could, in principle, skip
    re-walking the site. There are four ways to do that and only one is safe:

    1. *Seed the frontier with prior URLs.* Breaks the project's central claim.
       The agent would hop to a page it never discovered this run -- that is a
       small pre-built index, however it is described.
    2. *Pass them as ``exclude_urls``.* Actively wrong. "And the fees for that?"
       usually needs to re-read the very page the last turn ended on.
    3. *Cache pages across turns.* Breaks "every task starts live from the
       homepage" outright, and serves stale content besides.
    4. *A ranking bonus only.* Safe: a link still has to be discovered this run
       by following links from the seed, the model still chooses it, and the
       hop and page caps still bound the run. It is the same class of signal as
       the existing page-region weights.

    Only (4) is implemented, and it ships disabled
    (:data:`agent.config.FOLLOW_UP_PATH_BONUS` is ``0.0``). It is safe, but it
    buys roughly ten seconds of re-walking at the cost of complicating the
    clearest sentence in the project -- "no pre-built index, every task
    navigates live from the homepage" -- and it would pull a follow-up that is
    really a topic switch back toward the previous topic. Set the constant
    above zero to enable it; it is gated on ``used_context`` so a self-contained
    task is never affected, and it is logged whenever it changes an ordering.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Iterable

from agent import config
from agent.llm import LLMClient, LLMError
from browsing.extract_links import canonical_key

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from agent.navigator import NavigationResult

logger = logging.getLogger(__name__)

RESOLVER_SCHEMA = {
    "type": "object",
    "properties": {
        "sub_goal": {"type": "string"},
        "used_context": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": ["sub_goal", "used_context", "reasoning"],
    "additionalProperties": False,
}

RESOLVER_SYSTEM_PROMPT = """You rewrite a user's latest request into a single \
self-contained instruction, using the earlier turns of the conversation only \
where the latest request depends on them.

Rules:
- Do NOT answer the request. Rewrite it.
- If the latest request refers back to something earlier ("that one", "the \
second", "its fees"), replace the reference with what it refers to, taken from \
the earlier turns shown.
- If the latest request already stands on its own, return it essentially \
unchanged and set used_context to false. A new topic must not be contaminated \
by earlier turns.
- Never invent a name, product or detail that does not appear in the earlier \
turns or in the request itself. If a reference cannot be resolved from what is \
shown, keep the request as it is and say so in reasoning.
- Keep the rewrite short -- one sentence.

Reply with JSON only:
{"sub_goal": "<standalone request>", "used_context": <bool>, "reasoning": "<why>"}"""


# --------------------------------------------------------------------------
# What a turn remembers
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class HopSummary:
    """One navigation step, reduced to what a later turn might need."""

    url: str
    label: str | None
    source: str | None
    reasoning: str


@dataclass(frozen=True)
class TurnSummary:
    """A finished navigation, without any page content.

    Deliberately not a ``NavigationResult``: that holds a ``PageDict`` with the
    raw HTML and full text of the page reached.
    """

    status: str
    page_url: str | None
    final_reasoning: str
    sources: tuple[str, ...]
    hops: tuple[HopSummary, ...]

    @classmethod
    def from_loop(cls, result: Any) -> "TurnSummary":
        """Summarise a whole plan as one conversational turn.

        A turn is what the user asked, not what the planner did with it, so a
        multi-sub-goal run still records as one turn. ``status`` reports
        whether anything resolved, which is what a follow-up needs to know.
        """
        resolved = [o for o in result.outcomes if o.status == "resolved"]
        return cls(
            status="resolved" if resolved else "exhausted",
            page_url=resolved[0].source_url if resolved else None,
            final_reasoning=(resolved[0].reason if resolved
                             else (result.budget_exhausted or "nothing resolved")),
            sources=tuple(result.visited_urls),
            hops=tuple(
                HopSummary(url=step.url, label=step.label, source=step.source,
                           reasoning=step.reasoning)
                for step in result.trail
            ),
        )

    @classmethod
    def from_result(cls, result: "NavigationResult") -> "TurnSummary":
        return cls(
            status=result.status,
            page_url=result.page["url"] if result.page else None,
            final_reasoning=result.final_reasoning,
            sources=tuple(result.sources),
            hops=tuple(
                HopSummary(
                    url=step.url, label=step.label, source=step.source, reasoning=step.reasoning
                )
                for step in result.trail
            ),
        )


@dataclass(frozen=True)
class Resolution:
    """What the resolver made of a request."""

    sub_goal: str
    used_context: bool
    reasoning: str
    error: str | None = None


@dataclass(frozen=True)
class Turn:
    task: str
    resolved_sub_goal: str
    used_context: bool
    summary: TurnSummary
    timestamp: str


@dataclass
class Session:
    """One conversation. In memory, for this process only."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    turns: list[Turn] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_used: float = field(default_factory=time.monotonic)
    max_turns: int = config.SESSION_MAX_TURNS
    # Two turns of one conversation can run at once -- the API allows a second
    # task on the same session_id before the first finishes -- and both would
    # otherwise append and trim the turn list concurrently.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def touch(self) -> None:
        self.last_used = time.monotonic()

    def add_turn(self, task: str, resolution: Resolution, result: "NavigationResult") -> Turn:
        """Record a finished navigation, keeping only its summary."""
        with self._lock:
            return self._add_turn(task, resolution, result)

    def _add_turn(self, task: str, resolution: Resolution, result: "NavigationResult") -> Turn:
        turn = Turn(
            task=task,
            resolved_sub_goal=resolution.sub_goal,
            used_context=resolution.used_context,
            summary=(TurnSummary.from_loop(result) if hasattr(result, "outcomes")
                     else TurnSummary.from_result(result)),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.turns.append(turn)
        # Oldest turns fall out rather than growing the prompt without bound.
        if len(self.turns) > self.max_turns:
            del self.turns[: len(self.turns) - self.max_turns]
        self.touch()
        return turn

    def snapshot(self) -> tuple[Turn, ...]:
        """The turns so far, taken atomically."""
        with self._lock:
            return tuple(self.turns)

    def visited_keys(self) -> frozenset[str]:
        """Canonical keys of pages reached in earlier turns.

        Only ever used as a ranking hint, and only when
        :data:`agent.config.FOLLOW_UP_PATH_BONUS` is non-zero. See the module
        docstring for why that ships disabled.
        """
        return frozenset(
            canonical_key(url) for turn in self.snapshot() for url in turn.summary.sources
        )

    def context_lines(self) -> list[str]:
        """The conversation so far, compact enough to put in a prompt."""
        lines: list[str] = []
        for index, turn in enumerate(self.snapshot(), 1):
            lines.append(f'Turn {index} request: "{turn.task}"')
            if turn.resolved_sub_goal != turn.task:
                lines.append(f"  navigated for: {turn.resolved_sub_goal}")
            lines.append(f"  outcome: {turn.summary.status}")
            for hop in turn.summary.hops:
                if hop.label:
                    lines.append(f"  visited: {hop.label} ({hop.url})")
                else:
                    lines.append(f"  visited: {hop.url}")
        return lines


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------
def build_resolver_prompt(task: str, session: Session) -> str:
    lines = ["Earlier turns in this conversation:", *session.context_lines(), ""]
    lines.append(f'Latest request: "{task}"')
    lines.append("")
    lines.append("Rewrite the latest request as a single self-contained instruction.")
    return "\n".join(lines)


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_resolution(raw: str, task: str) -> Resolution:
    """Read the resolver's reply, falling back to the raw task on anything odd.

    A resolver that fails must never block a run: the raw task is always a
    usable sub-goal, just a less precise one.
    """
    text = _FENCE_RE.sub("", (raw or "").strip())
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("expected a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return _fallback(task, f"could not parse the resolver's reply ({exc})")
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as inner:
            return _fallback(task, f"could not parse the resolver's reply ({inner})")

    sub_goal = str(parsed.get("sub_goal") or "").strip()
    if not sub_goal:
        return _fallback(task, "the resolver returned an empty sub-goal")
    if len(sub_goal) > config.RESOLVER_MAX_SUB_GOAL_CHARS:
        # A model asked to rewrite a question sometimes answers it instead; the
        # giveaway is length. Falling back is safer than navigating for an essay.
        return _fallback(
            task,
            f"the resolver returned {len(sub_goal)} characters, which reads as an "
            f"answer rather than a rewrite",
        )

    return Resolution(
        sub_goal=sub_goal,
        used_context=_coerce_flag(parsed.get("used_context")),
        reasoning=str(parsed.get("reasoning") or "").strip() or "no explanation given",
    )


def _coerce_flag(value: object) -> bool:
    """Read a boolean the model may have sent as a string.

    bool("false") is True, so a model that answers with the word rather than
    the literal would have inverted the flag.
    """
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def _fallback(task: str, why: str) -> Resolution:
    logger.warning("session resolution fell back to the raw request: %s", why)
    return Resolution(
        sub_goal=task,
        used_context=False,
        reasoning=f"Using the request as written: {why}.",
        error=why,
    )


def resolve_task(task: str, session: Session | None, llm: LLMClient) -> Resolution:
    """Turn a possibly-referential request into a standalone sub-goal."""
    cleaned = (task or "").strip()
    if not cleaned:
        return Resolution(sub_goal="", used_context=False, reasoning="empty request")

    if session is None or not session.turns:
        # Nothing to resolve against, and no reason to spend a model call.
        logger.info(
            "session resolver: task=%r -> passed through (no prior turns in this session)",
            cleaned,
        )
        return Resolution(
            sub_goal=cleaned,
            used_context=False,
            reasoning="First request in this session; nothing to resolve against.",
        )

    prompt = build_resolver_prompt(cleaned, session)
    try:
        raw = llm.complete(prompt, system=RESOLVER_SYSTEM_PROMPT, schema=RESOLVER_SCHEMA)
    except LLMError as exc:
        return _fallback(cleaned, f"the resolver could not be reached ({exc})")

    resolution = parse_resolution(raw, cleaned)
    # Logged on every path, including the uninteresting ones. Gating this on
    # used_context meant the one case worth debugging -- the model deciding a
    # follow-up was self-contained, so the UI shows nothing -- was the only
    # case that produced no log line at all.
    logger.info(
        "session resolver: task=%r -> sub_goal=%r used_context=%s changed=%s (%s)",
        cleaned,
        resolution.sub_goal,
        resolution.used_context,
        resolution.sub_goal != cleaned,
        resolution.reasoning,
    )
    return resolution


# --------------------------------------------------------------------------
# Session storage
# --------------------------------------------------------------------------
class SessionStore:
    """In-memory sessions, bounded by age and count.

    Nothing here is persisted. The bounds exist because a server left running
    would otherwise accumulate sessions -- and their turn summaries -- forever.
    """

    def __init__(
        self,
        *,
        ttl_s: float = config.SESSION_TTL_S,
        max_sessions: int = config.SESSION_MAX_SESSIONS,
    ) -> None:
        self._sessions: dict[str, Session] = {}
        self._ttl_s = ttl_s
        self._max_sessions = max_sessions

    def create(self) -> Session:
        self._sweep()
        session = Session()
        self._sessions[session.session_id] = session
        if len(self._sessions) > self._max_sessions:
            oldest = min(self._sessions.values(), key=lambda item: item.last_used)
            del self._sessions[oldest.session_id]
            logger.info("evicted least-recently-used session %s", oldest.session_id)
        return session

    def get(self, session_id: str) -> Session | None:
        self._sweep()
        session = self._sessions.get(session_id)
        if session is not None:
            session.touch()
        return session

    def _sweep(self) -> None:
        cutoff = time.monotonic() - self._ttl_s
        for session_id in [
            key for key, value in self._sessions.items() if value.last_used < cutoff
        ]:
            del self._sessions[session_id]

    def __len__(self) -> int:
        return len(self._sessions)

    def ids(self) -> Iterable[str]:
        return tuple(self._sessions)
