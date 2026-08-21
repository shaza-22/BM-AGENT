"""
HTTP interface to the navigation agent.

What it does
    Exposes starting a navigation, watching it live, reading its result, and
    holding a conversation across several turns. Serves the single-page
    frontend from the same origin.

Inputs
    ``POST /api/tasks`` with ``{task, session_id?}``; ``GET`` for status,
    the SSE stream, and health.

Outputs
    JSON per :mod:`api.schemas`, and an SSE stream of ``resolved`` / ``hop`` /
    ``done`` / ``error`` events.

Why it is needed
    The assignment requires a usable interface with visible progress. A run
    takes 5-30 seconds, so the API starts work and streams it rather than
    holding a request open and showing a blank screen.

Deliberately thin
    This is transport. Navigation lives in ``agent/navigator.py``, memory in
    ``agent/session.py``, concurrency and the WAF-safe rate limiting in
    ``api/runner.py``. Nothing decides anything here.

Scope
    One task runs one sub-goal. Planning a task into several sub-goals,
    validating a page, extracting fields and synthesising an answer belong to
    the other half of the project; the validator is injected through
    ``app.state.registry`` so it drops in without touching a route, and every
    hop event already carries the ``sub_goal`` it belongs to.

Running it
    uvicorn api.app:app --host 127.0.0.1 --port 8000

    Bound to localhost on purpose. The domain allow-list means this cannot be
    pointed at another host, but an exposed endpoint would still let anyone
    spend the day's model quota and drive traffic at a WAF-protected bank from
    your address.
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from agent.session import SessionStore
from api.events import HEARTBEAT, sse
from api.runner import HEARTBEAT_S, TaskRegistry
from api.schemas import (
    HealthResponse,
    NewSessionResponse,
    TaskAccepted,
    TaskRequest,
    TaskStatus,
)

logger = logging.getLogger(__name__)

FRONTEND = pathlib.Path(__file__).resolve().parent.parent / "frontend" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Only shutdown needs a hook: each stream captures its own event loop when
    # it subscribes, so nothing depends on startup having run.
    try:
        yield
    finally:
        app.state.registry.shutdown()


def create_app(
    registry: TaskRegistry | None = None, sessions: SessionStore | None = None
) -> FastAPI:
    """Build the app. Injectable so tests can supply a stubbed registry."""
    app = FastAPI(title="Banque Misr research assistant", lifespan=lifespan)
    app.state.registry = registry or TaskRegistry()
    app.state.sessions = sessions or SessionStore()

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        provider, model, configured = app.state.registry.llm_description()
        return HealthResponse(
            status="ok",
            provider=provider,
            model=model,
            llm_configured=configured,
            active_runs=app.state.registry.active_runs,
            queued_runs=app.state.registry.queued_runs,
            sessions=len(app.state.sessions),
        )

    @app.post("/api/sessions", response_model=NewSessionResponse)
    def new_session() -> NewSessionResponse:
        session = app.state.sessions.create()
        return NewSessionResponse(session_id=session.session_id, created_at=session.created_at)

    @app.post("/api/tasks", response_model=TaskAccepted, status_code=202)
    def start_task(request: TaskRequest) -> TaskAccepted:
        if request.session_id:
            session = app.state.sessions.get(request.session_id)
            if session is None:
                # Sessions do not survive a restart. Saying so lets the UI
                # explain why follow-ups stopped working, where silently
                # starting a new one would leave the user guessing.
                raise HTTPException(
                    status_code=404,
                    detail=(
                        "unknown session: it expired or the server restarted. "
                        "Start a new conversation."
                    ),
                )
        else:
            session = app.state.sessions.create()

        record = app.state.registry.submit(request.task.strip(), session)
        return TaskAccepted(
            task_id=record.task_id, session_id=session.session_id, state=record.state
        )

    @app.get("/api/tasks/{task_id}", response_model=TaskStatus)
    def task_status(task_id: str) -> TaskStatus:
        record = app.state.registry.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown task")
        return record.to_status()

    @app.get("/api/tasks/{task_id}/stream")
    async def stream(task_id: str, request: Request) -> StreamingResponse:
        registry = app.state.registry
        record = registry.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown task")

        async def generate() -> AsyncIterator[str]:
            subscriber = registry.subscribe(record)
            try:
                while True:
                    if await request.is_disconnected():
                        return
                    try:
                        event = await asyncio.wait_for(
                            subscriber.queue.get(), timeout=HEARTBEAT_S
                        )
                    except asyncio.TimeoutError:
                        # Quiet stretches happen: a hop can take 30s. The
                        # comment keeps proxies from closing the connection.
                        yield HEARTBEAT
                        continue
                    yield sse(event.name, event.data)
                    if event.name in ("done", "error"):
                        return
            finally:
                registry.unsubscribe(record, subscriber)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # nginx would otherwise buffer the stream
            },
        )

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND)

    return app


app = create_app()
