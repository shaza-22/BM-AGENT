"""
Navigation layer for the Banque Misr agentic research assistant.

What it does
    Turns a sub-goal into a route through the live site: ranks the links the
    browsing layer found, asks a language model which one to follow, and loops
    until the sub-goal resolves or a cap stops it.

Inputs / Outputs
    See the module docstrings in ``navigator.py``, ``link_selector.py`` and
    ``llm.py``.

Why it is needed
    This is the agentic half of the project. The browsing layer can read any
    page; this layer decides which page to read next, and records why.
"""

from __future__ import annotations

"""Import-time bootstrap: see _bootstrap.py for why this is here."""

import pathlib as _pathlib
import sys as _sys

# Resolved from this file, not the working directory, so the server can be
# launched from anywhere. Guarded so a missing repo root cannot raise at import.
_repo_root = str(_pathlib.Path(__file__).resolve().parent.parent)
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)
try:
    from _bootstrap import ensure_src_on_path as _ensure_src_on_path

    _ensure_src_on_path()
except Exception:  # pragma: no cover - never let a path helper break an import
    pass


from typing import Any

_EXPORTS = {
    "SEED_URL": "agent.config",
    "PROVIDER": "agent.config",
    "LLMClient": "agent.llm",
    "LLMError": "agent.llm",
    "ClaudeLLMClient": "agent.llm",
    "GeminiLLMClient": "agent.llm",
    "FakeLLMClient": "agent.llm",
    "make_llm_client": "agent.llm",
    "Candidate": "agent.link_selector",
    "Selection": "agent.link_selector",
    "SelectionContext": "agent.link_selector",
    "rank_candidates": "agent.link_selector",
    "select_next_link": "agent.link_selector",
    "Navigator": "agent.navigator",
    "NavigationResult": "agent.navigator",
    "TrailStep": "agent.navigator",
    "navigate": "agent.navigator",
    "always_unresolved": "agent.navigator",
    "StepLogger": "agent.trail_log",
    "Resolution": "agent.session",
    "Session": "agent.session",
    "SessionStore": "agent.session",
    "Turn": "agent.session",
    "TurnSummary": "agent.session",
    "resolve_task": "agent.session",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(_EXPORTS[name]), name)


def __dir__() -> list[str]:
    return __all__
