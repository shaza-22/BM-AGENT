"""Integration tests for extraction against all real Banque Misr live fixtures."""

from pathlib import Path
from person_b.adapters.fixture_loader import FixtureLoader
from person_b.extraction.extractor import extract_content
from person_b.extraction.pdf_tables import extract_pdf_tables


def test_fixture_cards_category(fixture_loader: FixtureLoader):
    ctx = fixture_loader.load_text("home-smes-retail-banking-pages-cards.txt")
    res = extract_content(ctx)

    assert res.status == "success"
    entities = [e["name"] for e in res.extracted.get("entities", [])]
    assert "Credit Card" in entities
    assert "Debit cards" in entities
    assert len(res.evidence) >= 5


def test_fixture_credit_cards_list(fixture_loader: FixtureLoader):
    ctx = fixture_loader.load_text("home-smes-retail-banking-pages-cards-credit-cards-list.txt")
    res = extract_content(ctx)

    assert res.status == "success"
    cards = [e["name"] for e in res.extracted.get("entities", [])]
    assert "Classic Credit Card" in cards
    assert "Gold Credit Card" in cards
    assert "Titanium Credit Card" in cards
    assert "Visa Infinite" in cards
    assert len(cards) >= 10


def test_fixture_classic_credit_card_details(fixture_loader: FixtureLoader):
    ctx = fixture_loader.load_text("home-smes-retail-banking-pages-cards-credit-cards-pages-classic-credit-cards.txt")
    res = extract_content(ctx)

    assert res.status == "success"
    tables = res.extracted.get("tables", [])
    assert len(tables) == 3

    # Check Usage limits table
    limits_table = next(t for t in tables if "limits" in t["table_name"].lower())
    assert len(limits_table["rows"]) >= 8

    # Check Fees and charges table
    fees_table = next(t for t in tables if "fees" in t["table_name"].lower())
    assert len(fees_table["rows"]) >= 15
    issuance_row = next(r for r in fees_table["records"] if r.get("Fees and charges") == "Issuance")
    assert "250" in issuance_row.get("Details", "")

    # Check Benefits section
    benefits = res.extracted.get("sections", {}).get("Benefits", [])
    assert len(benefits) >= 10

    # Check evidence lineage
    assert len(res.evidence) >= 100
    for ev in res.evidence:
        assert ev.field is not None
        assert ev.value is not None
        assert ev.location is not None


def test_fixture_accounts_and_deposits_misleading_title(fixture_loader: FixtureLoader):
    """Proves body extraction succeeds despite misleading Classic Credit Card title on line 1."""
    ctx = fixture_loader.load_text("home-smes-retail-banking-accounts-and-deposits.txt")
    res = extract_content(ctx)

    assert res.status == "success"
    entities = [e["name"] for e in res.extracted.get("entities", [])]
    assert "Current Accounts" in entities
    assert "Savings Accounts" in entities
    assert "Time Deposits" in entities
    assert "Certificates of deposit" in entities


def test_fixture_consumer_loans(fixture_loader: FixtureLoader):
    ctx = fixture_loader.load_text("home-smes-retail-banking-consumer-loans.txt")
    res = extract_content(ctx)

    assert res.status == "success"
    loans = [e["name"] for e in res.extracted.get("entities", [])]
    assert "Personal loans" in loans
    assert "Mortgage Loan" in loans
    assert "Auto Loan" in loans
    assert "Solar panel loan" in loans


def test_fixture_fees_hub(fixture_loader: FixtureLoader):
    ctx = fixture_loader.load_text("home-pages-fees.txt")
    res = extract_content(ctx)

    assert res.status == "success"
    attachments = [e["name"] for e in res.extracted.get("entities", []) if e.get("type") == "attachment"]
    assert any("payment cards Fees" in a for a in attachments)
    assert any("Deposits Interest Rates" in a for a in attachments)


def test_fixture_fee_pdf_table_integrity(fixture_loader: FixtureLoader):
    """Verify cross-card PDF table extraction preserves card-to-column mappings."""
    pdf_ctx = fixture_loader.load_pdf("usage-limits-and-fees-en.pdf")
    res = extract_content(pdf_ctx)

    assert res.status == "success"
    pdf_tables = res.extracted.get("pdf_tables", [])
    assert len(pdf_tables) >= 5
    assert res.extracted.get("total_pdf_pages") == 7

    # Check that Table 1 on page 1 has card columns intact
    table1 = pdf_tables[0]
    headers = table1["headers"]
    assert "Classic" in headers
    assert "Gold" in headers

    # Verify evidence has page and table indices
    pdf_ev = [ev for ev in res.evidence if ev.location and ev.location.page_number is not None]
    assert len(pdf_ev) >= 100
    assert pdf_ev[0].location.page_number >= 1


def test_fixture_image_only_pdf_graceful_handling(fixture_loader: FixtureLoader):
    """Verify image-only ATM PDF returns unreadable status without crashing."""
    pdf_ctx = fixture_loader.load_pdf("media-guide-to-activate-debit-and-pre-paid-cards-through-the-atm-ashx.pdf")
    res = extract_content(pdf_ctx)

    assert res.status == "unreadable"
    assert res.extracted == {}
    assert len(res.warnings) >= 1
    assert "unreadable" in res.warnings[0].lower() or "scanned" in res.warnings[0].lower()
