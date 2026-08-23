"""Unit tests for PersonBConfig."""

import os
from person_b.config import PersonBConfig


def test_default_config():
    cfg = PersonBConfig()
    assert cfg.reasoning_provider == "fake"
    assert cfg.strict_verification is True
    assert cfg.confidence_threshold == 0.7
    assert cfg.max_expansion_sub_goals == 20


def test_env_override(monkeypatch):
    monkeypatch.setenv("PERSON_B_REASONING_PROVIDER", "gemini")
    monkeypatch.setenv("PERSON_B_CONFIDENCE_THRESHOLD", "0.85")
    monkeypatch.setenv("PERSON_B_STRICT_VERIFICATION", "false")
    monkeypatch.setenv("PERSON_B_REASONING_API_KEY", "secret-key-12345")

    cfg = PersonBConfig.from_env()
    assert cfg.reasoning_provider == "gemini"
    assert cfg.confidence_threshold == 0.85
    assert cfg.strict_verification is False

    # Check safe serialization masks key
    d = cfg.to_dict()
    assert d["reasoning_api_key"] == "***MASKED***"
