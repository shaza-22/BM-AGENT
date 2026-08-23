"""Robustness and edge case regression tests for Person B pipeline."""

from person_b.extraction.extractor import extract_content
from person_b.extraction.text_tables import parse_text_tables
from person_b.logging.jsonl_logger import PersonBLogger
from person_b.models import PageContext, SubGoal
from person_b.validation.validator import validate


def test_empty_and_whitespace_page_content():
    sg = SubGoal(id="sg1", question="Find fees")

    res_empty = validate(sg, "")
    assert res_empty["resolved"] is False
    assert res_empty["extracted"] == {}

    res_ws = validate(sg, "   \n\t   \n  ")
    assert res_ws["resolved"] is False
    assert res_ws["extracted"] == {}


def test_missing_source_url_handling():
    ctx = PageContext(content="Classic Credit Card\nFees and charges | Details\nIssuance | 250 EGP", source_url=None)
    res = extract_content(ctx)
    assert res.status == "success"
    assert len(res.evidence) > 0
    assert res.evidence[0].source_url is None


def test_unknown_content_type_fallback():
    ctx = PageContext(content="Some plain text body", content_type="application/octet-stream")
    res = extract_content(ctx)
    assert res.status in ("success", "empty")


def test_content_type_override_contradicting_is_pdf_hint():
    """If Person A passes text content but metadata says is_pdf=True, content_type wins."""
    ctx = PageContext(
        content="Credit Card Overview\nBenefits\n- Reward points",
        content_type="text/plain",
        metadata={"is_pdf": True},
    )
    res = extract_content(ctx)
    assert res.status == "success"
    assert "sections" in res.extracted


def test_malformed_pipe_rows():
    malformed_text = """
Col A | Col B | Col C
Val 1 | Val 2
Val 3 | Val 4 | Val 5 | Val 6
|
"""
    tables = parse_text_tables(malformed_text)
    assert len(tables) == 1
    # Parser should not crash on uneven columns
    assert len(tables[0]["rows"]) >= 1


def test_mixed_arabic_and_english_text():
    arabic_text = """
????? ?????? ???? ???
????? ?????? ??????????
Fees and charges | Details
?????? ??????? | 250 ????
"""
    res = extract_content(arabic_text)
    assert res.status == "success"
    assert len(res.extracted.get("tables", [])) == 1


def test_logger_error_resilience(tmp_path):
    """Logger write errors must not crash execution unless strict mode is enabled."""
    invalid_logger = PersonBLogger(log_dir=str(tmp_path / "read_only_test"), enabled=True, strict_logging=False)
    ev = invalid_logger.log("test_event", {"key": "val"})
    assert ev is not None
    assert ev.event_type == "test_event"
