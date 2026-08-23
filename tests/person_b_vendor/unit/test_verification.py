"""Unit tests for source attribution, claim verification, and finalization."""

from person_b.models import Claim, ClaimStatus
from person_b.verification.attribution import attribute_sources
from person_b.verification.self_check import finalize, validate_answer


def test_attribution_supported_and_unvisited_sources():
    visited = ["https://banquemisr.com/classic", "https://banquemisr.com/cards"]

    c1 = Claim(
        id="c1",
        statement="Classic issuance fee is 250 EGP",
        source_url="https://banquemisr.com/classic",
    )
    c2 = Claim(
        id="c2",
        statement="Unvisited card fee is 999 EGP",
        source_url="https://banquemisr.com/unvisited_page",
    )
    c3 = Claim(
        id="c3",
        statement="No source URL claim",
        source_url=None,
    )

    checks = attribute_sources([c1, c2, c3], visited_pages=visited, strict=True)
    assert len(checks) == 3
    assert checks[0]["status"] == ClaimStatus.SUPPORTED.value
    assert checks[1]["status"] == ClaimStatus.UNSUPPORTED.value
    assert checks[2]["status"] == ClaimStatus.UNSUPPORTED.value


def test_validate_answer_rejects_overbroad_claims():
    visited = ["https://banquemisr.com/classic"]
    c_overbroad = Claim(
        id="c_ob",
        statement="Classic Credit Card is the cheapest card in the bank",
        source_url="https://banquemisr.com/classic",
    )

    val_res = validate_answer([c_overbroad], visited_pages=visited)
    assert val_res["all_supported"] is False
    assert len(val_res["failed"]) == 1
    assert "overbroad" in val_res["failed"][0]["reason"].lower()


def test_finalize_removes_failed_claims():
    draft = {
        "draft_answer": "Draft answer text",
        "sources": ["https://banquemisr.com/classic"],
        "missing_info": [],
    }
    verification_res = {
        "passed": [
            {
                "claim": {"id": "c1", "statement": "Supported statement", "source_url": "https://banquemisr.com/classic"},
                "status": "supported",
                "verified_source_url": "https://banquemisr.com/classic",
            }
        ],
        "failed": [
            {
                "claim": {"id": "c2", "statement": "Unsupported statement"},
                "status": "unsupported",
            }
        ],
    }

    final = finalize(draft, verification_res)
    assert "https://banquemisr.com/classic" in final["source_urls"]
    assert len(final["verified_claims"]) == 1
