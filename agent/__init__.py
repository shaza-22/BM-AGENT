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
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(_EXPORTS[name]), name)


def __dir__() -> list[str]:
    return __all__
