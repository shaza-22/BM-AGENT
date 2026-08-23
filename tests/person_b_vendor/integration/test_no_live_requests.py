"""Verify that running Person B tests makes NO live network requests."""

import socket
import pytest
from person_b import plan_task, validate
from person_b.adapters.fixture_loader import FixtureLoader


def test_no_live_network_access(fixture_loader: FixtureLoader, monkeypatch):
    """Monkeypatch socket connect to fail immediately if network access is attempted."""
    def disallowed_connect(*args, **kwargs):
        pytest.fail("Live network request attempted by Person B code!")

    monkeypatch.setattr(socket.socket, "connect", disallowed_connect)

    # 1. Load offline fixture
    ctx = fixture_loader.load_text("home-smes-retail-banking-pages-cards-credit-cards-pages-classic-credit-cards.txt")

    # 2. Plan task
    plan = plan_task("Find fees for Classic card")

    # 3. Validate
    verdict = validate(plan.sub_goals[0], ctx)

    assert verdict["resolved"] is False
