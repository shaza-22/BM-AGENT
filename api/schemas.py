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
    sub_goal: str
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


class HealthResponse(BaseModel):
    status: Literal["ok"]
    provider: str
    model: str
    # Whether a key is present -- never the key, and no call is made to check.
    llm_configured: bool
    active_runs: int
    queued_runs: int
    sessions: int
