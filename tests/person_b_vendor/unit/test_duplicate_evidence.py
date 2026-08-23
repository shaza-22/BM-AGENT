"""Unit and integration tests for duplicate and conflicting evidence preservation."""

from person_b.extraction.extractor import extract_content
from person_b.models import Evidence, EvidenceLocation, ExtractionLocationType, PageContext


def test_duplicate_and_conflicting_evidence_detection():
    """Verify conflicting values across tables/prose are preserved with warnings."""
    text = """
Fees and charges
Fees and charges | Details
Issuance | EGP 250
Renewal | EGP 250

Other Fees Table
Fees and charges | Details
Issuance | EGP 300
"""
    res = extract_content(text, source_url="https://banquemisr.com/card")

    # Both evidence records must be preserved
    issuance_ev = [ev for ev in res.evidence if ev.field == "Details" and ev.location and ev.location.row_index is not None]
    assert len(issuance_ev) >= 2

    # Verify conflict warning was surfaced
    assert len(res.warnings) >= 1
    assert any("conflicting values" in w.lower() for w in res.warnings)


def test_agreeing_duplicate_evidence_preservation():
    """Verify agreeing values across tables are preserved without false conflict."""
    text = """
Table 1
Fees and charges | Details
Issuance | EGP 250

Table 2
Fees and charges | Details
Issuance | EGP 250
"""
    res = extract_content(text, source_url="https://banquemisr.com/card")
    assert len(res.evidence) >= 2
    # No conflict warning when values agree
    assert not any("conflicting values" in w.lower() for w in res.warnings)
