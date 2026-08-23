"""Planner for decomposing user research tasks into structured sub-goals."""

import re
from typing import Any, Dict, List, Optional, Union
import uuid

from person_b.config import PersonBConfig, default_config
from person_b.extraction.normalization import normalize_whitespace
from person_b.models import (
    Goal,
    Plan,
    SubGoal,
    SubGoalStatus,
    TaskType,
    ValidationResult,
)


def _classify_task_type(task_text: str) -> TaskType:
    """Classify user task into standard TaskType enum based on intent keywords."""
    lower = task_text.lower()
    if any(w in lower for w in ("compare", "difference", "vs", "versus", "better than", "comparison")):
        return TaskType.COMPARISON
    elif any(w in lower for w in ("recommend", "best", "suggest", "which card", "should i get", "suitable for")):
        return TaskType.RECOMMENDATION
    elif any(w in lower for w in ("what is", "how much", "find fee", "issuance fee", "grace period", "limit of")):
        return TaskType.LOOKUP
    elif any(w in lower for w in ("all", "list", "every", "steps", "guide")):
        return TaskType.MULTI_HOP
    return TaskType.GENERAL


def _extract_requested_fields(task_text: str) -> List[str]:
    """Extract requested dimensions (fees, benefits, limits, etc.) from task prompt."""
    lower = task_text.lower()
    fields = []
    if "fee" in lower or "charge" in lower or "cost" in lower or "issuance" in lower or "renewal" in lower:
        fields.append("fees")
    if "benefit" in lower or "feature" in lower or "reward" in lower or "discount" in lower:
        fields.append("benefits")
    if "limit" in lower or "withdrawal" in lower or "purchasing" in lower:
        fields.append("limits")
    if "interest" in lower or "installment" in lower or "rate" in lower:
        fields.append("interest_rate")
    if "eligibility" in lower or "requirement" in lower or "document" in lower:
        fields.append("eligibility")
    if not fields:
        if "card" in lower or "compare" in lower:
            fields = ["fees", "benefits"]
        else:
            fields = ["overview"]
    return fields


def plan_task(user_task: str, config: Optional[PersonBConfig] = None) -> Plan:
    """
    Decompose a raw user research task string into an initial structured Plan.

    Produces the minimal useful initial sub-goals without eagerly hardcoding entities
    before they are discovered from evidence.
    """
    cfg = config or default_config
    norm_task = normalize_whitespace(user_task)
    goal_id = f"g_{uuid.uuid4().hex[:8]}"
    plan_id = f"plan_{uuid.uuid4().hex[:8]}"

    task_type = _classify_task_type(norm_task)
    target_fields = _extract_requested_fields(norm_task)

    goal = Goal(
        id=goal_id,
        statement=norm_task,
        original_task=norm_task,
        task_type=task_type.value,
        constraints=[],
        metadata={"target_fields": target_fields},
    )

    lower = norm_task.lower()
    sub_goals: List[SubGoal] = []

    # 1. Broad / Discovery / Comparison tasks
    if task_type in (TaskType.COMPARISON, TaskType.RECOMMENDATION) or ("credit card" in lower and any(w in lower for w in ("all", "list", "compare", "cards"))):
        # Start with discovery sub-goal
        sg1 = SubGoal(
            id="sg_001",
            question="Find the list of Banque Misr credit cards",
            status=SubGoalStatus.PENDING,
            target_fields=["credit_cards_list"],
            created_from="initial_plan",
        )
        sub_goals.append(sg1)

    # 2. Loans discovery / lookup
    elif "loan" in lower:
        sg1 = SubGoal(
            id="sg_001",
            question=f"Find Banque Misr loan offerings and details for: {norm_task}",
            status=SubGoalStatus.PENDING,
            target_fields=target_fields,
            created_from="initial_plan",
        )
        sub_goals.append(sg1)

    # 3. Accounts & Deposits lookup
    elif "account" in lower or "deposit" in lower or "certificate" in lower:
        sg1 = SubGoal(
            id="sg_001",
            question=f"Find Banque Misr account and deposit details for: {norm_task}",
            status=SubGoalStatus.PENDING,
            target_fields=target_fields,
            created_from="initial_plan",
        )
        sub_goals.append(sg1)

    # 4. Direct product / specific topic lookup
    else:
        sg1 = SubGoal(
            id="sg_001",
            question=f"Find information for: {norm_task}",
            status=SubGoalStatus.PENDING,
            target_fields=target_fields,
            created_from="initial_plan",
        )
        sub_goals.append(sg1)

    return Plan(
        id=plan_id,
        goal=goal,
        sub_goals=sub_goals,
        version=1,
    )


def next_pending_sub_goal(plan: Plan) -> Optional[SubGoal]:
    """Retrieve the next pending sub-goal from the plan, if any."""
    for sg in plan.sub_goals:
        if sg.status == SubGoalStatus.PENDING:
            return sg
    return None


def apply_validation(
    plan: Plan,
    sub_goal: Union[str, SubGoal],
    verdict: Union[Dict[str, Any], ValidationResult],
) -> Plan:
    """Update sub-goal status and plan state based on validation outcome."""
    sg_id = sub_goal.id if isinstance(sub_goal, SubGoal) else sub_goal
    resolved = verdict.resolved if isinstance(verdict, ValidationResult) else verdict.get("resolved", False)

    for sg in plan.sub_goals:
        if sg.id == sg_id:
            if resolved:
                sg.status = SubGoalStatus.RESOLVED
            else:
                status_str = (
                    verdict.status.value
                    if isinstance(verdict, ValidationResult)
                    else verdict.get("status", "unresolved")
                )
                if status_str == "partial":
                    sg.status = SubGoalStatus.PARTIAL
                elif status_str == "unreadable":
                    sg.status = SubGoalStatus.NOT_AVAILABLE
                else:
                    sg.status = SubGoalStatus.PENDING
            break

    return plan


def mark_not_available(
    plan: Plan,
    sub_goal: Union[str, SubGoal],
    reason: str = "",
) -> Plan:
    """Mark a sub-goal as not available when search limits are exhausted."""
    sg_id = sub_goal.id if isinstance(sub_goal, SubGoal) else sub_goal
    for sg in plan.sub_goals:
        if sg.id == sg_id:
            sg.status = SubGoalStatus.NOT_AVAILABLE
            sg.metadata["not_available_reason"] = reason
            break
    return plan
