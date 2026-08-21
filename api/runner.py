"""
Running navigations behind the HTTP layer.

What it does
    Accepts a task, resolves it against a session, runs the navigation on a
    worker thread, and publishes progress events to anyone streaming.

Inputs
    ``TaskRegistry.submit(task, session, llm_factory)``.

Outputs
    A ``TaskRecord`` whose ``events`` queue feeds the SSE stream and whose
    accumulated ``status`` feeds the poll fallback.

Why it is needed
    A run takes 5-30 seconds and blocks (``requests`` and ``time.sleep``), so
    it cannot happen on the event loop, and the client cannot wait for it. This
    is the bridge: threads on one side, async streams on the other.

Rate limiting and the WAF (the load-bearing decision here)
    The site is behind an F5 WAF that bans on burst traffic, so two mechanisms
    are used together because neither is sufficient alone:

    * One :class:`~browsing.fetcher.RateLimiter` is shared by every run in the
      process. It keys per host under a lock, so however many runs are active,
      requests to banquemisr.com stay at most one per 1-2s. A concurrency cap
      alone would not do this -- two runs at one fetch per 1.5s each is one
      fetch per 0.75s, twice the intended rate.
    * A worker pool of :data:`MAX_CONCURRENT_RUNS` threads bounds how many runs
      queue behind that limiter. With a shared limiter and no cap, ten
      concurrent runs interleave their fetches and all ten take ten times as
      long; with a cap the third request waits its turn and is told so.

    Each run still gets its own ``Fetcher``, because the per-run cache and the
    visited-set belong to one task and must not be shared.

Lifecycle
    Everything is in memory. Records expire ``TASK_TTL_S`` after they finish.
    A client disconnecting does not cancel a run -- it completes and waits in
    the registry, so a reloaded page can still poll it. **If the server
    restarts, every task id, session and in-flight run is lost**; a poll for a
    pre-restart id returns 404, which the frontend reports as a lost run rather
    than a missing one.
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from agent import config as agent_config
from agent.llm import LLMClient, LLMError, make_llm_client
from agent.navigator import Navigator, ValidateFn
from agent.session import Resolution, Session, resolve_task
from agent.trail_log import StepLogger
from api.replay import REPLAY_EVENT_DELAY_S, find_recording, save_run
from browsing import config as browsing_config
from browsing.language import detect_language
from api.schemas import DoneEvent, ErrorEvent, HopEvent, ResolvedEvent, TaskStatus
from browsing.fetcher import Fetcher, RateLimiter

logger = logging.getLogger(__name__)

MAX_CONCURRENT_RUNS = 2
TASK_TTL_S = 1800.0
# Sent so a stream held open through a quiet stretch is not closed by a proxy.
HEARTBEAT_S = 15.0


@dataclass
class Event:
    """One thing worth telling the client about."""

    name: str
    data: dict[str, Any]


@dataclass
class Subscriber:
    """A live stream, together with the loop that is awaiting it.

    The loop is captured when the subscription is made rather than configured
    up front. An ``asyncio.Queue`` is not thread-safe: calling ``put_nowait``
    on it from a worker thread stores the item but does not reliably wake the
    coroutine waiting on ``get()``, so the stream stalls until a heartbeat and
    then stalls again. Every hand-off therefore goes through
    ``call_soon_threadsafe`` on the loop that actually owns the queue.
    """

    queue: asyncio.Queue
    loop: asyncio.AbstractEventLoop | None


@dataclass
class TaskRecord:
    task_id: str
    session_id: str
    task: str
    language: str | None = None
    replayed: bool = False
    state: str = "queued"
    queue_position: int | None = None
    sub_goal: str | None = None
    used_context: bool = False
    resolution_reasoning: str | None = None
    hops: list[HopEvent] = field(default_factory=list)
    result: DoneEvent | None = None
    error: ErrorEvent | None = None
    finished_at: float | None = None
    # Every event in order, so a client that connects late still sees the whole
    # run rather than only what happens from now on.
    history: list[Event] = field(default_factory=list)
    _subscribers: list["Subscriber"] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def terminal(self) -> bool:
        return self.state in ("done", "error")

    def to_status(self) -> TaskStatus:
        return TaskStatus(
            task_id=self.task_id,
            session_id=self.session_id,
            state=self.state,  # type: ignore[arg-type]
            task=self.task,
            sub_goal=self.sub_goal,
            language=self.language,
            replayed=self.replayed,
            used_context=self.used_context,
            resolution_reasoning=self.resolution_reasoning,
            queue_position=self.queue_position,
            hops=list(self.hops),
            result=self.result,
            error=self.error,
        )


class TaskRegistry:
    """Holds runs in flight and their results, for this process only."""

    def __init__(
        self,
        *,
        llm_factory: Callable[[], LLMClient] = make_llm_client,
        validate_fn: ValidateFn | None = None,
        fetcher_factory: Callable[[RateLimiter], Fetcher] | None = None,
        max_workers: int = MAX_CONCURRENT_RUNS,
        ttl_s: float = TASK_TTL_S,
        record_dir: pathlib.Path | None = None,
        replay_dir: pathlib.Path | None = None,
        replay_delay_s: float = REPLAY_EVENT_DELAY_S,
    ) -> None:
        self._records: dict[str, TaskRecord] = {}
        self._llm_factory = llm_factory
        self._validate_fn = validate_fn
        self._fetcher_factory = fetcher_factory or (lambda limiter: Fetcher(rate_limiter=limiter))
        self._ttl_s = ttl_s
        # Recording is always safe; replaying replaces navigation entirely and
        # is only ever on when a demo asks for it.
        self._record_dir = record_dir
        self._replay_dir = replay_dir
        self._replay_delay_s = replay_delay_s
        self._lock = threading.Lock()
        self._active = 0
        # Shared across every run in the process: this is what keeps the
        # request rate to the site constant no matter how many tasks are going.
        self._rate_limiter = RateLimiter()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="navrun")
        self._max_workers = max_workers

    # -- lifecycle ---------------------------------------------------------
    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    @property
    def active_runs(self) -> int:
        with self._lock:
            return self._active

    @property
    def queued_runs(self) -> int:
        with self._lock:
            return sum(1 for record in self._records.values() if record.state == "queued")

    def get(self, task_id: str) -> TaskRecord | None:
        self._sweep()
        return self._records.get(task_id)

    def _sweep(self) -> None:
        cutoff = time.monotonic() - self._ttl_s
        with self._lock:
            stale = [
                task_id
                for task_id, record in self._records.items()
                if record.finished_at is not None and record.finished_at < cutoff
            ]
            for task_id in stale:
                del self._records[task_id]

    # -- submission --------------------------------------------------------
    def submit(self, task: str, session: Session, language: str | None = None) -> TaskRecord:
        self._sweep()
        record = TaskRecord(
            task_id=uuid.uuid4().hex[:12],
            session_id=session.session_id,
            task=task,
            # An explicit choice wins; otherwise the task's own script decides,
            # which costs nothing and keeps a model request for the navigation.
            language=language or detect_language(task, default=browsing_config.LANGUAGE),
        )
        with self._lock:
            self._records[record.task_id] = record
            waiting = sum(1 for item in self._records.values() if item.state == "queued")
        record.queue_position = max(0, waiting - 1)
        if record.queue_position:
            self._publish(record, "queued", {"position": record.queue_position})
        self._pool.submit(self._run, record, session)
        return record

    # -- the run itself ----------------------------------------------------
    def _run(self, record: TaskRecord, session: Session) -> None:
        with self._lock:
            self._active += 1
        record.state = "running"
        record.queue_position = None
        try:
            if self._replay_dir is not None and self._replay(record):
                return
            llm = self._llm_factory()
            resolution = self._resolve(record, session, llm)
            self._navigate(record, session, llm, resolution)
        except Exception as exc:  # a bug here must not kill the worker thread
            logger.exception("run %s failed", record.task_id)
            self._fail(record, "internal", f"{type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self._active -= 1
            if not record.terminal:  # defensive: never leave a stream hanging
                self._fail(record, "internal", "the run ended without a result")
            if self._record_dir is not None and not record.replayed:
                save_run(record.task, record.language, record.task_id,
                         record.history, self._record_dir)

    def _replay(self, record: TaskRecord) -> bool:
        """Play a recorded run back through the live event channel.

        Returns False when there is nothing to replay, so the caller falls
        through to a real navigation.
        """
        recording = find_recording(self._replay_dir, record.task)
        if recording is None:
            logger.warning("replay mode is on but %s holds no recordings", self._replay_dir)
            return False

        record.replayed = True
        for event in recording.get("events", []):
            name, data = event.get("name"), dict(event.get("data") or {})
            if name == "resolved":
                # Marked as a replay on the way out; a recording must never be
                # presented as a live run.
                data["replayed"] = True
                record.sub_goal = data.get("sub_goal")
                record.used_context = bool(data.get("used_context"))
                record.resolution_reasoning = data.get("reasoning")
                record.language = data.get("language") or record.language
            elif name == "hop":
                record.hops.append(HopEvent(**data))
            elif name == "done":
                record.result = DoneEvent(**data)
                record.state = "done"
                record.finished_at = time.monotonic()
            elif name == "error":
                record.error = ErrorEvent(**data)
                record.state = "error"
                record.finished_at = time.monotonic()
            self._publish(record, name, data)
            if self._replay_delay_s:
                time.sleep(self._replay_delay_s)   # so it reads like a live run
        if not record.terminal:
            self._fail(record, "internal", "the recording ended without a result")
        return True

    def _resolve(self, record: TaskRecord, session: Session, llm: LLMClient) -> Resolution:
        resolution = resolve_task(record.task, session, llm)
        record.sub_goal = resolution.sub_goal
        record.used_context = resolution.used_context
        record.resolution_reasoning = resolution.reasoning
        self._publish(
            record,
            "resolved",
            ResolvedEvent(
                session_id=session.session_id,
                task=record.task,
                sub_goal=resolution.sub_goal,
                used_context=resolution.used_context,
                reasoning=resolution.reasoning,
                language=record.language,
            ).model_dump(),
        )
        return resolution

    def _navigate(
        self, record: TaskRecord, session: Session, llm: LLMClient, resolution: Resolution
    ) -> None:
        sub_goal = resolution.sub_goal

        def on_record(entry: dict[str, Any]) -> None:
            if entry.get("event") != "step":
                return
            hop = HopEvent(
                hop=entry["hop"],
                sub_goal=sub_goal,
                url=entry["url"],
                label=entry.get("label"),
                source=entry.get("source"),
                reasoning=entry.get("reasoning", ""),
                confidence=entry.get("confidence", 0.0),
                status=entry.get("fetch_status", 0),
                content_type=entry.get("content_type", "other"),
                validated=entry.get("validated", False),
                validate_reason=entry.get("validate_reason", ""),
                links_found=entry.get("links_found", 0),
                candidates_offered=entry.get("candidates_offered", 0),
                candidates_available=entry.get("candidates_available", 0),
                elapsed_ms=entry.get("elapsed_ms", 0),
            )
            record.hops.append(hop)
            self._publish(record, "hop", hop.model_dump())

        navigator = Navigator(
            llm,
            fetcher=self._fetcher_factory(self._rate_limiter),
            validate_fn=self._validate_fn,
            step_logger=StepLogger(on_record=on_record, run_id=record.task_id),
            language=record.language,
        )

        try:
            result = navigator.navigate(
                sub_goal,
                familiar_keys=session.visited_keys() if resolution.used_context else frozenset(),
            )
        except LLMError as exc:
            self._fail(record, "llm", str(exc))
            return

        session.add_turn(record.task, resolution, result)

        if result.status == "error":
            self._fail(record, "llm", result.final_reasoning or "the model could not be reached")
            return
        if result.status == "blocked":
            self._fail(record, "blocked", result.final_reasoning or "the site blocked the request")
            return

        done = DoneEvent(
            status=result.status,
            language=result.language,
            cap_hit=result.cap_hit,
            final_reasoning=result.final_reasoning,
            page_url=result.page["url"] if result.page else None,
            sources=list(result.sources),
            hops_used=result.hops_used,
            pages_fetched=result.pages_fetched,
            extracted=result.extracted,
            timing=(result.stats or {}).get("timing", {}),
        )
        record.result = done
        record.state = "done"
        record.finished_at = time.monotonic()
        self._publish(record, "done", done.model_dump())

    def _fail(self, record: TaskRecord, kind: str, message: str) -> None:
        if record.terminal:
            return
        record.error = ErrorEvent(kind=kind, message=message)  # type: ignore[arg-type]
        record.state = "error"
        record.finished_at = time.monotonic()
        self._publish(record, "error", record.error.model_dump())

    # -- fan-out -----------------------------------------------------------
    def _publish(self, record: TaskRecord, name: str, data: dict[str, Any]) -> None:
        event = Event(name=name, data=data)
        with record._lock:
            record.history.append(event)
            subscribers = list(record._subscribers)
        for subscriber in subscribers:
            self._deliver(subscriber, event)

    @staticmethod
    def _deliver(subscriber: Subscriber, event: Event) -> None:
        """Hand an event from a worker thread to the loop awaiting the queue."""
        loop = subscriber.loop
        if loop is None or not loop.is_running():
            # Nothing is awaiting from another thread, so the queue is safe to
            # touch directly (a synchronous test, or a stream already gone).
            subscriber.queue.put_nowait(event)
            return
        try:
            loop.call_soon_threadsafe(subscriber.queue.put_nowait, event)
        except RuntimeError:
            # The loop closed between the check and the call: the subscriber is
            # gone, and a departed client is not a reason to fail a run.
            logger.debug("dropped %s for a closed stream", event.name)

    def subscribe(self, record: TaskRecord) -> Subscriber:
        """Attach to a run, receiving everything it has already emitted first."""
        try:
            loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        subscriber = Subscriber(queue=asyncio.Queue(), loop=loop)
        with record._lock:
            for event in record.history:
                subscriber.queue.put_nowait(event)
            record._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, record: TaskRecord, subscriber: Subscriber) -> None:
        with record._lock:
            if subscriber in record._subscribers:
                record._subscribers.remove(subscriber)

    # -- introspection for /health ----------------------------------------
    def llm_description(self) -> tuple[str, str, bool]:
        provider = agent_config.PROVIDER
        model = (
            agent_config.GEMINI_MODEL if provider == "gemini" else agent_config.CLAUDE_MODEL
        )
        import os

        from agent.llm import load_project_env

        load_project_env()
        env_var = (
            agent_config.GEMINI_API_KEY_ENV
            if provider == "gemini"
            else agent_config.ANTHROPIC_API_KEY_ENV
        )
        # Presence only. The value is never read into a response or a log.
        return provider, model, bool((os.environ.get(env_var) or "").strip())
