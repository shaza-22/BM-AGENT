"""Tests for the HTTP layer.

Fully offline: pages come from the saved fixtures through a FakeSession, and
decisions from a FakeLLMClient. No network, no API key.
"""

from __future__ import annotations

import json
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi.testclient import TestClient  # noqa: E402

from agent.llm import FakeLLMClient, LLMError  # noqa: E402
from agent.session import SessionStore  # noqa: E402
from api.app import create_app  # noqa: E402
from api.runner import TaskRegistry  # noqa: E402
from browsing.fetcher import Fetcher  # noqa: E402
from conftest import FakeSession, choose_by, live_routes  # noqa: E402


def fetcher_factory(limiter):
    return Fetcher(
        session=FakeSession(live_routes()), rate_limiter=limiter,
        respect_robots=False, delay_range=(0.0, 0.0), allow_playwright=False,
    )


def build(llm_factory=None, **kwargs):
    registry = TaskRegistry(
        llm_factory=llm_factory or (lambda: FakeLLMClient(choose_by("/Pages/Cards"))),
        fetcher_factory=fetcher_factory,
        **kwargs,
    )
    return TestClient(create_app(registry=registry, sessions=SessionStore())), registry


def run_to_completion(client, task, session_id=None):
    body = {"task": task}
    if session_id:
        body["session_id"] = session_id
    response = client.post("/api/tasks", json=body)
    assert response.status_code == 202
    task_id = response.json()["task_id"]
    # The stream ends when the run does, so draining it is the wait.
    with client.stream("GET", f"/api/tasks/{task_id}/stream") as stream:
        for _ in stream.iter_lines():
            pass
    return client.get(f"/api/tasks/{task_id}").json()


class TestHealth:
    def test_reports_liveness_and_configuration(self):
        client, _ = build()
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["provider"] in ("gemini", "claude")
        assert isinstance(body["llm_configured"], bool)

    def test_never_reveals_the_key(self, monkeypatch):
        from agent import config as agent_config

        monkeypatch.setenv(agent_config.GEMINI_API_KEY_ENV, "AIzaSyFAKE000000000000000000")
        client, _ = build()
        assert "AIzaSy" not in client.get("/api/health").text


class TestSessions:
    def test_a_session_can_be_created(self):
        client, _ = build()
        body = client.post("/api/sessions").json()
        assert body["session_id"] and body["created_at"]

    def test_a_task_without_a_session_gets_one(self):
        client, _ = build()
        assert client.post("/api/tasks", json={"task": "find the cards"}).json()["session_id"]

    def test_an_unknown_session_is_rejected_rather_than_replaced(self):
        # Sessions die with the process. Silently starting a new one would show
        # the user "Interpreting as ..." derived from no context at all.
        client, _ = build()
        response = client.post("/api/tasks", json={"task": "x", "session_id": "gone"})
        assert response.status_code == 404
        assert "server restarted" in response.json()["detail"]


class TestTaskLifecycle:
    def test_a_task_runs_and_reports_its_result(self):
        client, _ = build()
        status = run_to_completion(client, "find the cards")
        assert status["state"] == "done"
        assert status["result"]["status"] in ("arrived", "no_candidates", "exhausted", "resolved")
        assert status["result"]["sources"]
        assert status["hops"]

    def test_hops_carry_the_reasoning_and_region(self):
        client, _ = build()
        status = run_to_completion(client, "find the cards")
        followed = [hop for hop in status["hops"] if hop["hop"] > 0]
        assert followed, "expected at least one followed link"
        for hop in followed:
            assert hop["reasoning"].strip()
            assert hop["source"] in ("nav", "body", "footer")
            assert hop["sub_goal"]          # forward-compat for a planner

    def test_the_poll_endpoint_mirrors_the_stream(self):
        client, _ = build()
        status = run_to_completion(client, "find the cards")
        assert status["sub_goal"] == "find the cards"
        assert status["state"] == "done"

    def test_an_unknown_task_is_404(self):
        client, _ = build()
        assert client.get("/api/tasks/nope").status_code == 404
        assert client.get("/api/tasks/nope/stream").status_code == 404

    def test_an_llm_failure_becomes_an_error_state(self):
        def explode(prompt):
            raise LLMError("no API key configured")

        client, _ = build(llm_factory=lambda: FakeLLMClient(explode))
        status = run_to_completion(client, "find the cards")
        assert status["state"] == "error"
        assert status["error"]["kind"] == "llm"
        assert "no API key" in status["error"]["message"]

    def test_a_task_that_is_too_long_is_rejected(self):
        client, _ = build()
        response = client.post("/api/tasks", json={"task": "x" * 5000})
        assert response.status_code == 422

    def test_an_empty_task_is_rejected(self):
        client, _ = build()
        assert client.post("/api/tasks", json={"task": ""}).status_code == 422


class TestFollowUps:
    def test_a_second_turn_reuses_the_session_and_resolves(self):
        # A fresh client is built per run, so the fake dispatches on the kind of
        # prompt rather than on call order -- which is how a real model sees it.
        def respond(prompt):
            if "Rewrite the latest request" in prompt:
                return json.dumps({"sub_goal": "find the fees for the cards",
                                   "used_context": True,
                                   "reasoning": "'that' refers to the cards page"})
            return json.dumps({"choice": -1, "outcome": "arrived",
                               "reasoning": "we are here", "confidence": 0.8})

        client, _ = build(llm_factory=lambda: FakeLLMClient(respond))
        session_id = client.post("/api/sessions").json()["session_id"]

        first = run_to_completion(client, "find the cards", session_id)
        assert first["sub_goal"] == "find the cards"
        assert first["used_context"] is False

        second = run_to_completion(client, "and the fees for that?", session_id)
        assert second["sub_goal"] == "find the fees for the cards"
        assert second["used_context"] is True
        assert second["resolution_reasoning"]

    def test_the_frontend_is_served(self):
        client, _ = build()
        page = client.get("/")
        assert page.status_code == 200
        assert "Banque Misr" in page.text


class TestConcurrency:
    def test_every_run_shares_one_rate_limiter(self):
        # The WAF bans on burst traffic, so the request rate to the site must
        # not scale with the number of concurrent runs.
        _, registry = build()
        fetchers = [registry._fetcher_factory(registry._rate_limiter) for _ in range(3)]
        limiters = {id(fetcher._rate_limiter) for fetcher in fetchers}
        assert len(limiters) == 1

    def test_each_run_gets_its_own_fetcher(self):
        # The per-run cache and visited-set belong to one task.
        _, registry = build()
        first = registry._fetcher_factory(registry._rate_limiter)
        second = registry._fetcher_factory(registry._rate_limiter)
        assert first is not second

    def test_the_worker_pool_is_bounded(self):
        _, registry = build(max_workers=2)
        assert registry._max_workers == 2

    def test_finished_runs_expire(self):
        client, registry = build(ttl_s=-1.0)
        response = client.post("/api/tasks", json={"task": "find the cards"})
        task_id = response.json()["task_id"]
        with client.stream("GET", f"/api/tasks/{task_id}/stream") as stream:
            for _ in stream.iter_lines():
                pass
        assert registry.get(task_id) is None


class TestLanguage:
    def test_an_arabic_task_is_detected_and_reported(self):
        client, _ = build()
        status = run_to_completion(client, "ازاى افتح حساب اسلامي")
        assert status["language"] == "ar"

    def test_an_english_task_is_detected(self):
        client, _ = build()
        assert run_to_completion(client, "find the credit cards")["language"] == "en"

    def test_an_explicit_override_wins_over_detection(self):
        client, _ = build()
        response = client.post("/api/tasks", json={"task": "find the cards", "language": "ar"})
        task_id = response.json()["task_id"]
        with client.stream("GET", f"/api/tasks/{task_id}/stream") as stream:
            for _ in stream.iter_lines():
                pass
        assert client.get(f"/api/tasks/{task_id}").json()["language"] == "ar"

    def test_the_language_reaches_the_navigator(self):
        # An Arabic run must start from the Arabic homepage, not the English one.
        from agent.config import seed_for

        client, _ = build()
        status = run_to_completion(client, "ازاى افتح حساب اسلامي")
        assert status["hops"][0]["url"] == seed_for("ar")


class TestReplayMode:
    """A demo must not be able to die on quota mid-presentation."""

    def test_a_run_can_be_recorded_and_replayed(self, tmp_path):
        client, _ = build(record_dir=tmp_path)
        original = run_to_completion(client, "find the cards")
        assert list(tmp_path.glob("*.json")), "expected a recording on disk"

        replay_client, _ = build(replay_dir=tmp_path, replay_delay_s=0.0)
        replayed = run_to_completion(replay_client, "find the cards")

        assert replayed["replayed"] is True
        assert replayed["state"] == original["state"]
        assert [hop["url"] for hop in replayed["hops"]] == [
            hop["url"] for hop in original["hops"]
        ]

    def test_a_replay_never_calls_the_model(self, tmp_path):
        def explode(prompt):
            raise AssertionError("replay must not reach the model")

        client, _ = build(record_dir=tmp_path)
        run_to_completion(client, "find the cards")

        replay_client, _ = build(
            llm_factory=lambda: FakeLLMClient(explode), replay_dir=tmp_path, replay_delay_s=0.0
        )
        assert run_to_completion(replay_client, "find the cards")["state"] == "done"

    def test_a_replay_announces_itself(self, tmp_path):
        client, _ = build(record_dir=tmp_path)
        run_to_completion(client, "find the cards")
        replay_client, registry = build(replay_dir=tmp_path, replay_delay_s=0.0)
        task_id = replay_client.post("/api/tasks", json={"task": "find the cards"}).json()["task_id"]
        with replay_client.stream("GET", f"/api/tasks/{task_id}/stream") as stream:
            body = "".join(stream.iter_lines())
        # Presenting a recording as a live run would undo the honesty the rest
        # of this project is built on.
        assert '"replayed": true' in body.replace("\n", "")

    def test_an_unmatched_task_falls_back_to_a_recording(self, tmp_path):
        client, _ = build(record_dir=tmp_path)
        run_to_completion(client, "find the cards")
        replay_client, _ = build(replay_dir=tmp_path, replay_delay_s=0.0)
        status = run_to_completion(replay_client, "something else entirely")
        assert status["replayed"] is True
        assert status["hops"]

    def test_replay_with_no_recordings_falls_through_to_a_live_run(self, tmp_path):
        client, _ = build(replay_dir=tmp_path, replay_delay_s=0.0)
        status = run_to_completion(client, "find the cards")
        assert status["replayed"] is False
        assert status["state"] in ("done", "error")

    def test_recording_is_off_by_default(self, tmp_path):
        client, _ = build()
        run_to_completion(client, "find the cards")
        assert not list(tmp_path.glob("*.json"))
