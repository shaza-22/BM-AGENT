"""
Server-Sent Events formatting.

What it does
    Turns an :class:`~api.runner.Event` into the SSE wire format, and provides
    the heartbeat comment.

Inputs / Outputs
    ``sse(event)`` -> the bytes-ready text for one event.

Why it is needed
    SSE framing is easy to get subtly wrong -- a missing blank line makes a
    stream that never delivers -- so it lives in one tested place.
"""

from __future__ import annotations

import json
from typing import Any

HEARTBEAT = ": ping\n\n"


def sse(name: str, data: Any) -> str:
    """One SSE event: a name, a single-line JSON payload, a blank line."""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {name}\ndata: {payload}\n\n"
