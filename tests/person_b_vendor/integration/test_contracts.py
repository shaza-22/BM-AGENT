"""Mandatory Person A ? Person B integration contract compatibility tests."""

from person_b import validate
from person_b.models import PageContext, SubGoal


def test_validate_mandatory_minimum_contract():
    """Verify validate(sub_goal, page_content) works with exactly two positional args."""
    sub_goal = "Find the issuance fee for Banque Misr Classic Credit Card"
    page_content = "Banquemisr - Classic Credit Card\nFees and charges | Details\nIssuance | EGP 250"

    result = validate(sub_goal, page_content)

    # Mandatory contract keys
    assert isinstance(result, dict)
    assert "resolved" in result
    assert "extracted" in result
    assert "reason" in result

    assert isinstance(result["resolved"], bool)
    assert isinstance(result["extracted"], dict)
    assert isinstance(result["reason"], str)


def test_validate_enriched_arguments_contract():
    """Verify validate() accepts all documented enriched keyword arguments."""
    sub_goal = SubGoal(
        id="sg_001",
        question="Find the issuance fee for Banque Misr Classic Credit Card",
        target_fields=["fees"],
        metadata={"target_entity": "Classic Credit Card"},
    )
    page_context = PageContext(
        content="Banquemisr - Classic Credit Card\nFees and charges | Details\nIssuance | EGP 250",
        source_url="https://www.banquemisr.com/classic",
        content_type="html",
        status_code=200,
        metadata={"fetch_mode": "requests"},
    )

    result = validate(
        sub_goal,
        page_context,
        source_url="https://www.banquemisr.com/classic",
        content_type="html",
        status_code=200,
        pdf_bytes=None,
        pdf_path=None,
        pdf_tables=None,
        metadata={"priority": "high"},
    )

    assert "resolved" in result
    assert "extracted" in result
    assert "reason" in result
    assert result.get("source_url") == "https://www.banquemisr.com/classic"
