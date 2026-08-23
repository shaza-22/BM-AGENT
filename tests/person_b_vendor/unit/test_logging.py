"""Unit tests for structured JSONL logging."""

import json
from person_b.logging.jsonl_logger import PersonBLogEvent, PersonBLogger


def test_log_event_serialization():
    ev = PersonBLogEvent(
        event_type="validation_attempted",
        run_id="run_001",
        sub_goal_id="sg_001",
        data={"source_url": "https://www.banquemisr.com/cards", "resolved": False},
    )
    json_str = ev.to_json()
    d = json.loads(json_str)
    assert d["event_type"] == "validation_attempted"
    assert d["run_id"] == "run_001"
    assert d["data"]["resolved"] is False


def test_logger_in_memory_accumulation(tmp_path):
    logger = PersonBLogger(log_dir=str(tmp_path), run_id="test_run")
    ev1 = logger.log("task_started", {"task": "Find cards"})
    ev2 = logger.log("plan_created", {"sub_goals_count": 1})

    assert len(logger._events) == 2
    logger.flush()
    assert len(logger._events) == 0

    log_file = tmp_path / "person_b_events.jsonl"
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
