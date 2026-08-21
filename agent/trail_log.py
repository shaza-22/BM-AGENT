"""
Structured step logging for the navigation agent.

What it does
    Emits one JSON object per navigation step and keeps them in memory for the
    caller.

Inputs
    ``StepLogger(stream=..., run_id=...)`` and ``emit(record)``.

Outputs
    JSON Lines on the given stream (one object per line), plus ``.records``.

Why it is needed
    The step log is a graded deliverable and feeds two consumers: the frontend
    progress feed and the evaluation pipeline. Both need to read it
    mechanically, so the format is JSON Lines rather than prose -- the
    human-readable version is the ``logging`` output, which is separate.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Callable, TextIO

logger = logging.getLogger(__name__)


class StepLogger:
    """Collects step records and optionally writes them as JSON Lines."""

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        run_id: str | None = None,
        on_record: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._stream = stream
        # Called with each record as it is emitted. The API uses this to push
        # progress to a live SSE stream; without it the runner would have to
        # serialise to JSON through a fake stream and parse it straight back.
        self._on_record = on_record
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.records: list[dict[str, Any]] = []

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        record = {"run_id": self.run_id, "event": event, **fields}
        self.records.append(record)
        if self._stream is not None:
            self._stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._stream.flush()
        if self._on_record is not None:
            try:
                self._on_record(record)
            except Exception:
                # A consumer failing must never break the navigation it is
                # watching -- the run is the deliverable, the stream is not.
                logger.exception("step logger callback failed for %r", event)
        logger.debug("%s %s", event, record)
        return record

    def as_jsonl(self) -> str:
        """Every record so far, as a JSON Lines string."""
        return "\n".join(
            json.dumps(record, ensure_ascii=False, default=str) for record in self.records
        )
