"""Unit tests for real validation engine and false-positive defenses."""

from person_b.adapters.fixture_loader import FixtureLoader
from person_b.models import SubGoal, ValidationStatus
from person_b.validation.validator import validate


def test_validation_resolves_card_discovery(fixture_loader: FixtureLoader):
    """Verify credit-card-list fixture resolves discovery sub-goal."""
    ctx = fixture_loader.load_text("home-smes-retail-banking-pages-cards-credit-cards-list.txt")
    sg = SubGoal(
        id="sg_001",
        question="Find the list of Banque Misr credit cards",
        target_fields=["credit_cards_list"],
    )

    res = validate(sg, ctx)
    assert res["resolved"] is True
    assert res["status"] == ValidationStatus.RESOLVED.value
    assert len(res["extracted"].get("entities", [])) >= 10


def test_validation_rejects_irrelevant_fixture(fixture_loader: FixtureLoader):
    """Verify consumer loans fixture does NOT resolve credit card fees sub-goal."""
    ctx = fixture_loader.load_text("home-smes-retail-banking-consumer-loans.txt")
    sg = SubGoal(
        id="sg_002",
        question="Find fees for Classic Credit Card",
        target_fields=["fees"],
        metadata={"target_entity": "Classic Credit Card"},
    )

    res = validate(sg, ctx)
    assert res["resolved"] is False
    assert res["status"] == ValidationStatus.UNRESOLVED.value


def test_validation_misleading_title_defense(fixture_loader: FixtureLoader):
    """Verify Accounts & Deposits page (which has Classic title on line 1) does NOT resolve Classic card fees."""
    ctx = fixture_loader.load_text("home-smes-retail-banking-accounts-and-deposits.txt")
    sg = SubGoal(
        id="sg_002",
        question="Find fees for Classic Credit Card",
        target_fields=["fees"],
        metadata={"target_entity": "Classic Credit Card"},
    )

    res = validate(sg, ctx)
    assert res["resolved"] is False, "Misleading line 1 title must not falsely resolve card sub-goal"

    # But Accounts & Deposits sub-goal SHOULD resolve against it:
    sg_accounts = SubGoal(
        id="sg_003",
        question="Find Banque Misr accounts and deposit types",
        target_fields=["accounts_list"],
    )
    res_accounts = validate(sg_accounts, ctx)
    assert res_accounts["resolved"] is True
    assert res_accounts["status"] == ValidationStatus.RESOLVED.value


def test_validation_partial_evidence(fixture_loader: FixtureLoader):
    """Verify page with benefits but missing requested loan/mortgage fields returns partial status."""
    ctx = fixture_loader.load_text("home-smes-retail-banking-pages-cards-credit-cards-pages-classic-credit-cards.txt")
    sg = SubGoal(
        id="sg_004",
        question="Find benefits and mortgage loan terms for Classic Credit Card",
        target_fields=["benefits", "mortgage_terms"],
        metadata={"target_entity": "Classic Credit Card"},
    )

    res = validate(sg, ctx)
    assert res["resolved"] is False
    assert res["status"] == ValidationStatus.PARTIAL.value
    assert "mortgage_terms" in res["missing_fields"]
    # Evidence for benefits must still be preserved
    assert len(res["evidence"]) > 0


def test_validation_unreadable_pdf(fixture_loader: FixtureLoader):
    """Verify unreadable image-only PDF returns unreadable status without crashing."""
    pdf_ctx = fixture_loader.load_pdf("media-guide-to-activate-debit-and-pre-paid-cards-through-the-atm-ashx.pdf")
    sg = SubGoal(
        id="sg_005",
        question="Find ATM card activation steps",
    )

    res = validate(sg, pdf_ctx)
    assert res["resolved"] is False
    assert res["status"] == ValidationStatus.UNREADABLE.value
    assert "unreadable" in res["reason"].lower() or "scanned" in res["reason"].lower()
