"""
Browsing (perception) layer for the Banque Misr agentic research assistant.

What it does
    Gives the agent its two senses: fetching a live page (``fetcher``) and
    seeing where it can go next (``extract_links``). Nothing in this package
    knows anything about products, categories or topics -- it is pure
    navigation machinery, so the same code serves any Banque Misr research
    task.

Inputs / Outputs
    See the module docstrings in ``fetcher.py`` and ``extract_links.py``.

Why it is needed
    The agent has no pre-built index and no crawler. Every task starts fresh at
    the homepage and navigates live, so the quality of these two modules sets
    the ceiling on what the planner above them can achieve.

Usage
    >>> from browsing import Fetcher, extract_links
    >>> fetcher = Fetcher()                     # one per task
    >>> page = fetcher.fetch("https://www.banquemisr.com/")
    >>> links = extract_links(page["raw_html"], page["url"])
"""

from __future__ import annotations

from typing import Any

# Re-exports are resolved lazily (PEP 562). Importing the submodules eagerly
# here would make `python -m browsing.fetcher` re-execute an already-imported
# module and warn about it, and the __main__ blocks are how these modules are
# exercised against fixtures.
_EXPORTS = {
    "LANGUAGE": "browsing.config",
    "Fetcher": "browsing.fetcher",
    "PageDict": "browsing.fetcher",
    "RateLimiter": "browsing.fetcher",
    "escalation_reason": "browsing.fetcher",
    "fetch_page": "browsing.fetcher",
    "html_to_text": "browsing.fetcher",
    "pdf_to_text": "browsing.fetcher",
    "robots_allowed": "browsing.fetcher",
    "LinkDict": "browsing.extract_links",
    "canonical_key": "browsing.extract_links",
    "extract_links": "browsing.extract_links",
    "link_label": "browsing.extract_links",
    "link_source": "browsing.extract_links",
    "normalize_url": "browsing.extract_links",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(_EXPORTS[name]), name)


def __dir__() -> list[str]:
    return __all__
