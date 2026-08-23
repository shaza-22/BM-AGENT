"""Tests for the SSE progress stream.

A run takes 5-30 seconds, so the stream is what stands between the user and a
blank screen. Its two guarantees: events arrive in a usable order, and the
stream always terminates.
"""

from __future__ import annotations

import json
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi.testclient import TestClient  # noqa: E402

from agent.llm import FakeLLMClient, LLMError  # noqa: E402
from agent.session import SessionStore  # noqa: E402
from api.events import sse  # noqa: E402
from api.app import create_app  # noqa: E402
from api.runner import TaskRegistry  # noqa: E402
from browsing.fetcher import Fetcher  # noqa: E402
from conftest import FakeSession, choose_by, live_routes  # noqa: E402


def fetcher_factory(limiter):
    return Fetcher(
        session=FakeSession(live_routes()), rate_limiter=limiter,
        respect_robots=False, delay_range=(0.0, 0.0), allow_playwright=False,
    )


def build(llm_factory=None):
    registry = TaskRegistry(
        llm_factory=llm_factory or (lambda: FakeLLMClient(choose_by("/Pages/Cards"))),
        fetcher_factory=fetcher_factory,
    )
    return TestClient(create_app(registry=registry, sessions=SessionStore()))


def collect(client, task="find the cards"):
    """Drain a run's stream into a list of (event_name, payload)."""
    task_id = client.post("/api/tasks", json={"task": task}).json()["task_id"]
    events: list[tuple[str, dict]] = []
    name = None
    with client.stream("GET", f"/api/tasks/{task_id}/stream") as stream:
        for line in stream.iter_lines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                events.append((name, json.loads(line.split(":", 1)[1])))
    return events


class TestFraming:
    def test_an_event_is_framed_correctly(self):
        # A missing blank line makes a stream that never delivers.
        text = sse("hop", {"hop": 1})
        assert text == 'event: hop\ndata: {"hop": 1}\n\n'

    def test_payloads_are_single_line(self):
        text = sse("done", {"reasoning": "line one\nline two"})
        body = text.split("data: ", 1)[1]
        assert body.count("\n") == 2      # end of data line, then the blank line

    def test_the_response_declares_the_right_media_type(self):
        client = build()
        task_id = client.post("/api/tasks", json={"task": "x"}).json()["task_id"]
        with client.stream("GET", f"/api/tasks/{task_id}/stream") as stream:
            assert stream.headers["content-type"].startswith("text/event-stream")
            for _ in stream.iter_lines():
                pass


class TestOrdering:
    def test_resolved_comes_first_and_done_last(self):
        names = [name for name, _ in collect(build())]
        assert names[0] == "resolved"
        assert names[-1] == "done"
        assert "hop" in names

    def test_hops_arrive_in_order_within_each_sub_goal(self):
        """Hop numbering restarts per sub-goal, and must climb within one.

        It used to be monotonic across the whole stream, because a task was
        exactly one navigation. A task is now a plan: each sub-goal walks live
        from the seed and numbers its own hops from 0, which is why every hop
        carries ``sub_goal`` and the UI groups by it. Asserting a single global
        sequence would now be asserting that plans do not exist.
        """
        by_sub_goal: dict[str, list[int]] = {}
        for name, data in collect(build()):
            if name == "hop":
                by_sub_goal.setdefault(data["sub_goal"], []).append(data["hop"])

        assert by_sub_goal, "no hops were streamed"
        for sub_goal, hops in by_sub_goal.items():
            assert hops == sorted(hops), f"{sub_goal!r} streamed hops out of order: {hops}"
            assert hops[0] == 0, f"{sub_goal!r} did not start at the seed"

    def test_the_seed_is_the_first_hop(self):
        first = next(data for name, data in collect(build()) if name == "hop")
        assert first["hop"] == 0
        assert first["url"].rstrip("/") == "https://www.banquemisr.com"

    def test_every_hop_carries_what_the_ui_needs(self):
        for name, data in collect(build()):
            if name != "hop":
                continue
            assert {"hop", "url", "reasoning", "confidence", "sub_goal"} <= set(data)
            assert data["reasoning"].strip()


class TestTermination:
    def test_a_successful_run_terminates_with_done(self):
        names = [name for name, _ in collect(build())]
        assert names.count("done") == 1
        assert "error" not in names

    def test_a_failing_run_terminates_with_error(self):
        def explode(prompt):
            raise LLMError("model unreachable")

        events = collect(build(llm_factory=lambda: FakeLLMClient(explode)))
        assert events[-1][0] == "error"
        assert events[-1][1]["kind"] == "llm"

    def test_an_unexpected_crash_still_terminates_the_stream(self):
        # A bug in the run must not leave a client waiting forever.
        def explode(prompt):
            raise RuntimeError("something unforeseen")

        events = collect(build(llm_factory=lambda: FakeLLMClient(explode)))
        assert events[-1][0] == "error"
        assert events[-1][1]["kind"] == "internal"

    def test_streams_never_end_without_a_terminal_event(self):
        for factory in [
            None,
            lambda: FakeLLMClient(lambda p: (_ for _ in ()).throw(LLMError("down"))),
            lambda: FakeLLMClient(lambda p: "not json at all"),
        ]:
            names = [name for name, _ in collect(build(llm_factory=factory))]
            assert names[-1] in ("done", "error"), names


class TestLateSubscribers:
    def test_a_client_connecting_late_receives_the_whole_run(self):
        # The run may finish before anyone subscribes; the history replay is
        # what makes the poll fallback and a reconnect equivalent.
        client = build()
        task_id = client.post("/api/tasks", json={"task": "find the cards"}).json()["task_id"]
        with client.stream("GET", f"/api/tasks/{task_id}/stream") as stream:
            for _ in stream.iter_lines():
                pass
        names = [name for name, _ in _drain(client, task_id)]
        assert names[0] == "resolved" and names[-1] in ("done", "error")


def _drain(client, task_id):
    events, name = [], None
    with client.stream("GET", f"/api/tasks/{task_id}/stream") as stream:
        for line in stream.iter_lines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                events.append((name, json.loads(line.split(":", 1)[1])))
    return events
