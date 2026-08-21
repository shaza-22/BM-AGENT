"""
Recording and replaying runs, so a demo cannot die on quota.

What it does
    Saves the event stream of a finished run to a JSON file, and plays a saved
    run back through the same SSE channel as a live one.

Inputs
    ``save_run(record, directory)`` after a run terminates;
    ``find_recording(directory, task)`` when replay mode is on.

Outputs
    ``<directory>/<task_id>.json`` holding the task, its language and every
    event in order. ``find_recording`` returns the best match, or ``None``.

Why it is needed
    The free tier allows about twenty model requests a day and a sub-goal costs
    three or four. A rehearsal plus a presentation can exceed that, and a
    demo that fails in the room because a quota reset at midnight is a bad way
    to find out. Replay serves a real recorded run -- the same events, the same
    order, the same reasoning text -- without spending a request.

Honesty
    A replayed run is announced as one: the stream carries ``replayed: true``
    on its ``resolved`` event and the UI shows a badge. Presenting a recording
    as a live run would be the kind of demo shortcut this project has avoided
    everywhere else.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

logger = logging.getLogger(__name__)

# Paced so a replayed run reads like a live one rather than appearing at once.
REPLAY_EVENT_DELAY_S = 0.35


def save_run(task: str, language: str | None, task_id: str, events: list[Any],
             directory: pathlib.Path) -> pathlib.Path | None:
    """Write one finished run to *directory*. Never raises."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{task_id}.json"
        path.write_text(
            json.dumps(
                {
                    "task": task,
                    "language": language,
                    "task_id": task_id,
                    "events": [{"name": event.name, "data": event.data} for event in events],
                },
                ensure_ascii=False,  # Arabic stays readable in the file
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        logger.info("recorded run %s to %s", task_id, path)
        return path
    except OSError as exc:
        # Recording is a convenience; failing to record must not fail the run.
        logger.warning("could not record run %s: %s", task_id, exc)
        return None


def load_recordings(directory: pathlib.Path) -> list[dict]:
    if not directory.is_dir():
        return []
    runs: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        try:
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("skipping unreadable recording %s: %s", path, exc)
    return runs


def find_recording(directory: pathlib.Path, task: str) -> dict | None:
    """The recording that best matches *task*.

    Exact text first, then a containment match either way, then whichever
    recording exists -- a demo should show something rather than nothing, and
    the stream says plainly that it is a replay.
    """
    runs = load_recordings(directory)
    if not runs:
        return None

    wanted = (task or "").strip().casefold()
    for run in runs:
        if (run.get("task") or "").strip().casefold() == wanted:
            return run
    for run in runs:
        recorded = (run.get("task") or "").strip().casefold()
        if recorded and (recorded in wanted or wanted in recorded):
            return run
    logger.info("no recording matches %r; replaying %r instead", task, runs[0].get("task"))
    return runs[0]
