"""
Request and response shapes for the HTTP layer.

What it does
    Defines the pydantic models FastAPI validates against, and the JSON shape
    of every event the progress stream emits.

Inputs / Outputs
    Used by ``api/app.py`` for route signatures and by ``api/runner.py`` to
    build event payloads.

Why it is needed
    Keeping the wire format in one file means the frontend has a single place
    to read, and the shapes stay stable while the layers underneath change.

Forward compatibility
    Every hop carries ``sub_goal``. Today a task always resolves to exactly one
    sub-goal, but the planner this project will grow produces several; when it
    lands, the stream carries hops for each and the UI groups by that field
    without any shape change here.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from agent import config

TaskState = Literal["queued", "running", "done", "error"]


class NewSessionResponse(BaseModel):
    session_id: str
    created_at: str


class TaskRequest(BaseModel):
    # A task string goes straight into a prompt, so it is bounded here rather
    # than discovered to be a problem at the model.
    task: str = Field(min_length=1, max_length=config.MAX_TASK_CHARS)
    session_id: str | None = None
    # Explicit override; detection from the task's script decides when absent.
    language: str | None = None


class TaskAccepted(BaseModel):
    task_id: str
    session_id: str
    state: TaskState


class HopEvent(BaseModel):
    hop: int
    # The sub-goal this hop was walked for -- the planner's question, not the
    # user's task, since a task is now a plan of several. Hop numbering
    # restarts at 0 for each, because each walks live from the seed; the id is
    # what the UI groups by.
    sub_goal: str
    sub_goal_id: str | None = None
    url: str
    label: str | None = None
    source: str | None = None
    reasoning: str
    confidence: float
    status: int
    content_type: str
    validated: bool
    validate_reason: str
    links_found: int
    candidates_offered: int
    candidates_available: int
    elapsed_ms: int


class ResolvedEvent(BaseModel):
    session_id: str
    task: str
    sub_goal: str
    used_context: bool
    reasoning: str
    language: str | None = None
    # True when the events that follow come from a recording rather than a
    # live navigation. Surfaced so a replay is never passed off as live.
    replayed: bool = False


# --- plan-level events -----------------------------------------------------
# Added when the intelligence layer landed. A task no longer maps to exactly
# one navigation: it becomes a plan of sub-goals, each navigated live from the
# seed, and the stream carries the plan's shape as well as the hops. Every hop
# already carried ``sub_goal``, so the UI groups by it without a shape change
# there.


class PlanSubGoal(BaseModel):
    id: str
    question: str
    status: str = "pending"
    # One line from the planner on why this sub-goal is needed. Empty when the
    # keyword planner produced it, which has no such notion.
    why: str = ""


class PlanEvent(BaseModel):
    plan_id: str
    task_type: str
    target_fields: list[str] = Field(default_factory=list)
    sub_goals: list[PlanSubGoal] = Field(default_factory=list)
    gate_enabled: bool = True
    # "keyword" -- the vendored planner's keyword branches. "model" -- decomposed
    # by agent/planner.py. Surfaced so a keyword match is never presented as
    # agentic planning.
    plan_source: str = "keyword"
    plan_reasoning: str = ""
    pages_used: int = 0
    pages_max: int = 0
    llm_calls_used: int = 0
    llm_calls_max: int = 0


class SubGoalStartedEvent(BaseModel):
    sub_goal_id: str
    question: str
    index: int
    total: int
    target_fields: list[str] = Field(default_factory=list)


class SubGoalFinishedEvent(BaseModel):
    sub_goal_id: str
    question: str
    status: str
    reason: str = ""
    source_url: str | None = None
    nav_status: str = ""
    hops_used: int = 0
    pages_fetched: int = 0
    gate: dict[str, Any] | None = None
    pages_used: int = 0
    pages_max: int = 0
    llm_calls_used: int = 0
    llm_calls_max: int = 0


class PlanExpandedEvent(BaseModel):
    parent_id: str | None = None
    added: list[PlanSubGoal] = Field(default_factory=list)
    new_total: int = 0


class GateEvent(BaseModel):
    """A resolve the acceptance gate withheld.

    Emitted rather than only logged because it changes what the run does: the
    validator said stop and the run kept going. A silent override is the kind
    of thing that makes a system impossible to reason about from outside.
    """

    url: str
    sub_goal_id: str | None = None
    validator_reason: str = ""
    accepted: bool = False
    reason: str = ""
    snippets_checked: int = 0
    snippets_unique: int = 0
    pages_compared: int = 0


class BudgetEvent(BaseModel):
    pages_used: int = 0
    pages_max: int = 0
    llm_calls_used: int = 0
    llm_calls_max: int = 0


class SynthesisStartedEvent(BaseModel):
    validated_results: int = 0
    visited_pages: int = 0


class AnswerVerifiedEvent(BaseModel):
    """The verification numbers, which are the visible evidence it ran.

    ``support_rate`` is 1.0 when there were no claims at all, which would read
    as "fully supported" -- so ``claims_total`` travels with it and the UI
    shows nothing rather than 100% when it is zero.
    """

    support_rate: float = 1.0
    claims_total: int = 0
    claims_supported: int = 0
    claims_unsupported: int = 0
    claims_contradicted: int = 0
    prose_removed: list[str] = Field(default_factory=list)


class AnswerComposedEvent(BaseModel):
    """Whether the final wording was written by the model, and what was struck.

    Emitted even when composition was declined, because *why* it was declined
    is the interesting case: "no verified claims" means the run refused to let
    a model write about a bank with nothing to go on.
    """

    attempted: bool = False
    used: bool = False
    reason: str = ""
    claims_offered: int = 0
    grounding: dict[str, Any] | None = None
    # Sentences by tier -- "fact", "analysis", "judgment". Carried separately
    # from the answer text so the interface can show which statements the
    # bank's website is responsible for and which the agent is.
    tiered: bool = False
    tiers: dict[str, list[str]] = Field(default_factory=dict)


class AnswerEvent(BaseModel):
    """The finished, verified answer."""

    # "template" -- assembled from extracted values by the vendored, model-free
    # synthesis. "composed" -- written by the model from the verified claims and
    # then put through agent/grounding.py. The interface says which, because the
    # two do not carry the same warranty.
    answer_source: str = "template"
    composition: AnswerComposedEvent | None = None
    answer: str = ""
    source_urls: list[str] = Field(default_factory=list)
    not_found: list[str] = Field(default_factory=list)
    support_rate: float = 1.0
    claims_total: int = 0
    claims_supported: int = 0
    resolved_count: int = 0
    sub_goals_total: int = 0
    budget_exhausted: str | None = None
    gate_rejections: int = 0


class DoneEvent(BaseModel):
    status: str
    language: str | None = None
    cap_hit: str | None = None
    final_reasoning: str = ""
    page_url: str | None = None
    sources: list[str] = Field(default_factory=list)
    hops_used: int = 0
    pages_fetched: int = 0
    extracted: dict[str, Any] | None = None
    timing: dict[str, Any] = Field(default_factory=dict)
    # Present once the loop is driving; absent for a bare navigation.
    answer: AnswerEvent | None = None


class ErrorEvent(BaseModel):
    kind: Literal["llm", "blocked", "internal"]
    message: str


class TaskStatus(BaseModel):
    """The poll fallback: everything the stream has emitted, accumulated."""

    task_id: str
    session_id: str
    state: TaskState
    task: str
    sub_goal: str | None = None
    language: str | None = None
    replayed: bool = False
    used_context: bool = False
    resolution_reasoning: str | None = None
    queue_position: int | None = None
    hops: list[HopEvent] = Field(default_factory=list)
    result: DoneEvent | None = None
    error: ErrorEvent | None = None
    # Plan state, accumulated the same way hops are, so a client that polls
    # instead of streaming sees the same picture.
    plan: PlanEvent | None = None
    sub_goals: list[SubGoalFinishedEvent] = Field(default_factory=list)
    gate_events: list[GateEvent] = Field(default_factory=list)
    verification: AnswerVerifiedEvent | None = None
    answer: AnswerEvent | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    provider: str
    model: str
    # Whether a key is present -- never the key, and no call is made to check.
    llm_configured: bool
    active_runs: int
    queued_runs: int
    sessions: int
