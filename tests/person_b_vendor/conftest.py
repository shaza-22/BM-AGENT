"""Pytest configuration and shared fixtures for Person B test suite."""

# VENDOR ADAPTATION (not a Person B bug fix): their conftest inserted
# ``<pkg>/src`` on sys.path and resolved fixtures at ``<pkg>/fixtures/live``.
# Both paths assumed the package was its own repo root. Here ``pythonpath`` in
# pyproject.toml puts src/ on the path, and the fixtures live at the repo's
# single ``fixtures/live`` (a superset of theirs -- it adds the Arabic pages).
from pathlib import Path

import pytest
from person_b.adapters.fixture_loader import FixtureLoader
from person_b.adapters.reasoning_backend import FakeReasoningBackend
from person_b.config import PersonBConfig


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to live fixtures directory."""
    return Path(__file__).resolve().parents[2] / "fixtures" / "live"


@pytest.fixture
def fixture_loader(fixtures_dir: Path) -> FixtureLoader:
    """Return FixtureLoader pointing to test fixtures."""
    return FixtureLoader(str(fixtures_dir))


@pytest.fixture
def fake_backend() -> FakeReasoningBackend:
    """Return clean FakeReasoningBackend instance."""
    backend = FakeReasoningBackend()
    backend.reset()
    return backend


@pytest.fixture
def test_config(fixtures_dir: Path) -> PersonBConfig:
    """Return PersonBConfig suited for offline testing."""
    return PersonBConfig(
        reasoning_provider="fake",
        fixtures_dir=str(fixtures_dir),
        strict_verification=True,
    )
