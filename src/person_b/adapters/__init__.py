"""Adapters for external providers and local fixture loading."""

from person_b.adapters.fixture_loader import FixtureLoader
from person_b.adapters.reasoning_backend import (
    FakeReasoningBackend,
    ReasoningBackend,
    get_reasoning_backend,
)

__all__ = [
    "ReasoningBackend",
    "FakeReasoningBackend",
    "get_reasoning_backend",
    "FixtureLoader",
]
