"""Unit tests for comparison, recommendation, and synthesis reasoning."""

from person_b.models import ComparisonItem, Evidence
from person_b.reasoning.compare import compare_items
from person_b.reasoning.recommend import recommend
from person_b.reasoning.synthesize import synthesize


def test_compare_items_preserves_missing_and_evidence():
    it1 = ComparisonItem(
        name="Classic Credit Card",
        attributes={"issuance_fee": "250 EGP", "grace_period": "56 days"},
        evidence=[Evidence(field="issuance_fee", value="250 EGP")],
    )
    it2 = ComparisonItem(
        name="Gold Credit Card",
        attributes={"issuance_fee": "500 EGP"},
        evidence=[Evidence(field="issuance_fee", value="500 EGP")],
    )

    res = compare_items([it1, it2])
    assert "differences" in res
    assert res["differences"]["issuance_fee"]["Classic Credit Card"] == "250 EGP"
    assert res["differences"]["issuance_fee"]["Gold Credit Card"] == "500 EGP"
    # Missing grace period in Gold must be explicitly Not Available
    assert res["differences"]["grace_period"]["Gold Credit Card"] == "Not Available"


def test_recommend_based_on_criteria():
    it1 = ComparisonItem(
        name="Classic Credit Card",
        attributes={"issuance_fee": "250 EGP"},
    )
    it2 = ComparisonItem(
        name="Platinum Card",
        attributes={"issuance_fee": "1000 EGP", "lounge_access": "Free airport lounge"},
    )

    # 1. Low fee criteria
    rec_fee = recommend({"priority": "lowest fee"}, [it1, it2])
    assert rec_fee["recommended_item"] == "Classic Credit Card"
    assert "250" in rec_fee["rationale"]

    # 2. Travel criteria
    rec_travel = recommend({"priority": "travel and airport"}, [it1, it2])
    assert rec_travel["recommended_item"] == "Platinum Card"


def test_synthesize_produces_claims():
    validated_results = [
        {
            "source_url": "https://banquemisr.com/cards",
            "extracted": {
                "entities": [{"name": "Classic Credit Card"}, {"name": "Gold Credit Card"}],
                "tables": [
                    {
                        "table_name": "Fees",
                        "records": [{"Issuance": "250 EGP"}],
                    }
                ],
            },
        }
    ]

    res = synthesize("Compare cards", validated_results=validated_results)
    assert len(res["claims"]) >= 1
    assert "https://banquemisr.com/cards" in res["sources"]
    assert res["claims"][0]["source_url"] == "https://banquemisr.com/cards"
