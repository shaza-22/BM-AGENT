"""Evidence-bound recommendation reasoning based on user criteria."""

from typing import Any, Dict, List, Optional, Union

from person_b.extraction.normalization import extract_currency_value
from person_b.models import ComparisonItem


def recommend(
    criteria: Dict[str, Any],
    items: List[Union[Dict[str, Any], ComparisonItem]],
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Generate recommendation matching user criteria against validated evidence.

    Separates factual premises from recommendation judgment.
    Exposes tradeoffs and missing evidence.
    """
    if not items:
        return {
            "recommended_item": None,
            "rationale": "No items provided for recommendation.",
            "factual_premises": [],
            "tradeoffs": [],
            "missing_evidence": ["No candidate products available"],
        }

    # Normalize items
    parsed_items = []
    for it in items:
        if isinstance(it, ComparisonItem):
            parsed_items.append({"name": it.name, "attributes": it.attributes, "evidence": it.evidence})
        elif isinstance(it, dict):
            parsed_items.append({"name": it.get("name", "Unknown"), "attributes": it.get("attributes", {}), "evidence": it.get("evidence", [])})

    priority = str(criteria.get("priority", criteria.get("goal", ""))).lower()

    recommended_name = None
    factual_premises = []
    tradeoffs = []
    missing_evidence = []

    # 1. Low fee priority
    if "fee" in priority or "cost" in priority or "low" in priority or "issuance" in priority:
        lowest_fee = float("inf")
        best_item = None

        for it in parsed_items:
            attrs = it["attributes"]
            fee_str = str(attrs.get("issuance_fee", attrs.get("annual_fee", attrs.get("fee", ""))))
            curr_info = extract_currency_value(fee_str)
            if curr_info:
                amount = curr_info["amount"]
                factual_premises.append(f"Fact: {it['name']} fee is {fee_str}.")
                if amount < lowest_fee:
                    lowest_fee = amount
                    best_item = it
            else:
                missing_evidence.append(f"Missing numeric fee evidence for {it['name']}.")

        if best_item:
            recommended_name = best_item["name"]
            rationale = f"Recommended '{recommended_name}' because it has the lowest documented fee ({lowest_fee} EGP)."
            for it in parsed_items:
                if it["name"] != recommended_name:
                    tradeoffs.append(f"'{it['name']}' has higher fee or higher limits compared to '{recommended_name}'.")
        else:
            recommended_name = parsed_items[0]["name"]
            rationale = f"Candidate '{recommended_name}' chosen as available option, but exact fee numbers were incomplete."

    # 2. Travel / International priority
    elif "travel" in priority or "international" in priority or "foreign" in priority:
        best_item = None
        for it in parsed_items:
            attrs = it["attributes"]
            if any(w in str(attrs).lower() for w in ("international", "travel", "usd", "airport", "lounge")):
                best_item = it
                factual_premises.append(f"Fact: {it['name']} has international/travel benefits documented in evidence.")
                break

        if best_item:
            recommended_name = best_item["name"]
            rationale = f"Recommended '{recommended_name}' based on documented international and travel features."
        else:
            recommended_name = parsed_items[0]["name"]
            rationale = f"'{recommended_name}' selected; explicit travel benefit comparison was limited across products."
            missing_evidence.append("Detailed travel lounge and foreign markup rates.")

    # 3. Default / General criteria
    else:
        recommended_name = parsed_items[0]["name"]
        factual_premises.append(f"Fact: Available options include {[it['name'] for it in parsed_items]}.")
        rationale = f"'{recommended_name}' is suitable for general banking needs."

    return {
        "recommended_item": recommended_name,
        "rationale": rationale,
        "factual_premises": factual_premises,
        "tradeoffs": tradeoffs,
        "missing_evidence": missing_evidence,
    }
