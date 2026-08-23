"""Structured comparison across items and products based strictly on validated evidence."""

from typing import Any, Dict, List, Optional, Union

from person_b.models import (
    ComparisonItem,
    ComparisonResult,
    Evidence,
)


def compare_items(
    items: List[Union[Dict[str, Any], ComparisonItem]],
    fields_to_compare: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Compare multiple products/entities strictly using available evidence.

    Preserves:
      - Explicit missing values (None / 'Not Available')
      - Attached Evidence provenance per field
      - Exact textual values, units, and conditions
      - Surfaces detected differences across items
    """
    normalized_items: List[ComparisonItem] = []
    for item in items:
        if isinstance(item, ComparisonItem):
            normalized_items.append(item)
        elif isinstance(item, dict):
            # Parse evidence objects if raw dicts
            ev_list = []
            for ev in item.get("evidence", []):
                if isinstance(ev, Evidence):
                    ev_list.append(ev)
                elif isinstance(ev, dict):
                    ev_list.append(Evidence.from_dict(ev))

            normalized_items.append(
                ComparisonItem(
                    name=item.get("name", "Unknown Item"),
                    attributes=item.get("attributes", {}),
                    evidence=ev_list,
                )
            )

    # Determine compared fields (union of attributes or explicit fields)
    if fields_to_compare:
        compared_fields = list(fields_to_compare)
    else:
        field_set = set()
        for it in normalized_items:
            field_set.update(it.attributes.keys())
        compared_fields = sorted(list(field_set)) if field_set else ["overview"]

    # Compute differences across items for each field
    differences: Dict[str, Dict[str, Any]] = {}
    for f in compared_fields:
        field_values = {}
        for it in normalized_items:
            val = it.attributes.get(f)
            field_values[it.name] = val if val is not None else "Not Available"
        differences[f] = field_values

    # Build summary
    summary_lines = [f"Comparison of {len(normalized_items)} items across {len(compared_fields)} fields:"]
    for f in compared_fields:
        vals = [f"{k}: {v}" for k, v in differences[f].items()]
        summary_lines.append(f"  - {f}: " + "; ".join(vals))

    result = ComparisonResult(
        items=normalized_items,
        compared_fields=compared_fields,
        differences=differences,
        summary="\n".join(summary_lines),
    )

    return result.to_dict()
