"""Dynamic plan expansion based on validated discovered entities."""

import re
from typing import Any, Dict, List, Optional, Set, Union

from person_b.config import PersonBConfig, default_config
from person_b.extraction.normalization import normalize_entity_name
from person_b.models import (
    ExtractionResult,
    Plan,
    SubGoal,
    SubGoalStatus,
    TaskType,
    ValidationResult,
)


def _extract_entity_names_from_payload(payload: Union[Dict[str, Any], ValidationResult, ExtractionResult]) -> List[str]:
    """Extract candidate entity names from extraction/validation result."""
    entities: List[str] = []

    if isinstance(payload, ValidationResult):
        data = payload.extracted or {}
    elif isinstance(payload, ExtractionResult):
        data = payload.extracted or {}
    elif isinstance(payload, dict):
        data = payload.get("extracted", payload)
    else:
        return []

    # 1. Check 'entities' list of dicts
    if "entities" in data and isinstance(data["entities"], list):
        for item in data["entities"]:
            if isinstance(item, dict) and item.get("name"):
                entities.append(item["name"])
            elif isinstance(item, str):
                entities.append(item)

    # 2. Check 'products' or 'cards' lists
    for key in ("products", "cards", "card_names", "loan_products"):
        if key in data and isinstance(data[key], list):
            for item in data[key]:
                if isinstance(item, str):
                    entities.append(item)
                elif isinstance(item, dict) and item.get("name"):
                    entities.append(item["name"])

    return [normalize_entity_name(e) for e in entities if e and isinstance(e, str)]


def expand_plan(
    plan: Plan,
    verdict: Union[Dict[str, Any], ValidationResult, ExtractionResult],
    config: Optional[PersonBConfig] = None,
) -> Plan:
    """
    Dynamically expand plan with new research sub-goals based on discovered entities.

    Features:
      - Deterministic sub-goal IDs
      - Robust deduplication preventing duplicate sub-goals
      - Respects max expansion limits
      - Preserves parent-child provenance
    """
    cfg = config or default_config
    discovered_entities = _extract_entity_names_from_payload(verdict)
    if not discovered_entities:
        return plan

    # Gather existing sub-goal queries and targeted entities for deduplication
    existing_queries: Set[str] = {sg.question.lower().strip() for sg in plan.sub_goals}
    existing_entities: Set[str] = set()
    for sg in plan.sub_goals:
        entity_meta = sg.metadata.get("target_entity")
        if entity_meta:
            existing_entities.add(normalize_entity_name(entity_meta).lower())

    target_fields = plan.goal.metadata.get("target_fields", ["fees", "benefits"])
    fields_desc = ", ".join(target_fields) if target_fields else "fees, benefits"

    # Identify triggering sub-goal if any
    parent_sub_goal_id = plan.sub_goals[0].id if plan.sub_goals else None

    next_idx = len(plan.sub_goals) + 1
    new_sub_goals: List[SubGoal] = []

    for entity in discovered_entities:
        norm_entity = normalize_entity_name(entity)
        if not norm_entity:
            continue

        norm_lower = norm_entity.lower()
        if norm_lower in existing_entities:
            continue

        # Check if question already exists
        question = f"Find {fields_desc} for {norm_entity}"
        if question.lower().strip() in existing_queries:
            continue

        # Check expansion cap
        if len(plan.sub_goals) + len(new_sub_goals) >= cfg.max_expansion_sub_goals:
            break

        sg = SubGoal(
            id=f"sg_{next_idx:03d}",
            question=question,
            status=SubGoalStatus.PENDING,
            target_fields=list(target_fields),
            parent_id=parent_sub_goal_id,
            created_from="dynamic_expansion",
            metadata={"target_entity": norm_entity},
        )
        new_sub_goals.append(sg)
        existing_queries.add(question.lower().strip())
        existing_entities.add(norm_lower)
        next_idx += 1

    if new_sub_goals:
        plan.sub_goals.extend(new_sub_goals)
        plan.version += 1

    return plan
