"""Unit tests for ReasoningBackend and FakeReasoningBackend."""

import pytest
from person_b.adapters.reasoning_backend import (
    FakeReasoningBackend,
    ReasoningBackend,
    get_reasoning_backend,
)
from person_b.config import PersonBConfig
from person_b.errors import AdapterError


def test_fake_backend_completion():
    backend = FakeReasoningBackend()
    backend.register_response("classify", "lookup")

    resp = backend.complete("Please classify this task: 'fees for classic'")
    assert resp == "lookup"

    # Verify history recorded
    assert len(backend.call_history) == 1
    assert backend.call_history[0]["method"] == "complete"


def test_fake_backend_structured():
    backend = FakeReasoningBackend()
    canned = {"task_type": "comparison", "entities": ["Classic", "Gold"]}
    backend.register_structured("compare", canned)

    resp = backend.structured_complete("compare Classic and Gold cards", schema={})
    assert resp == canned
    assert len(backend.call_history) == 1


def test_get_reasoning_backend_factory():
    cfg = PersonBConfig(reasoning_provider="fake")
    backend = get_reasoning_backend(cfg)
    assert isinstance(backend, FakeReasoningBackend)

    invalid_cfg = PersonBConfig(reasoning_provider="nonexistent_provider")
    with pytest.raises(AdapterError):
        get_reasoning_backend(invalid_cfg)
