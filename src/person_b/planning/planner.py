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
    if any(w in lower for w in ("compare", "difference", "vs", "versus", "better than", "comparison",
                                # --- PATCH 18 (vendor) ---------------------
                                # "Which is better, a personal loan or a car
                                # loan?" matched none of the originals --
                                # "better than" needs the word "than" -- so it
                                # fell through to GENERAL. Downstream, task
                                # type is what decides whether a plan may fan
                                # out to several entities, and this is a
                                # comparison that must.
                                "better", "which is worse", " or a ", " or an ")):
        return TaskType.COMPARISON
    elif any(w in lower for w in ("recommend", "best", "suggest", "should i get", "suitable for",
                                  # PATCH 18: "which card" was the only
                                  # category-specific token in this function
                                  # and only ever matched one product line.
                                  "which account", "which loan", "which one",
                                  "best for", "right for me")):
        return TaskType.RECOMMENDATION
    elif any(w in lower for w in ("what is", "how much", "find fee", "issuance fee", "grace period", "limit of")):
        return TaskType.LOOKUP
    # --- PATCH 18 (vendor) --------------------------------------------------
    # Was: any(w in lower for w in ("all", "list", "every", "steps", "guide")).
    # Bare "list" is a noun at least as often as a verb -- "where can I find the
    # branch list?" is a single lookup, and classifying it MULTI_HOP made the
    # plan fan out to every entity on the page it landed on. MULTI_HOP now
    # needs the enumerate-*and*-detail sense: several things, each with an
    # attribute, which is the only reading that genuinely needs several pages.
    elif any(w in lower for w in ("list all", "list every", "show me all", "show me every",
                                  "all the", "each of", "for each", "every one",
                                  "and their", "and its ", "steps", "guide")):
        return TaskType.MULTI_HOP
    # --- END PATCH 18 ---
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
    # --- PATCH 10 (vendor) --------------------------------------------------
    # Dropped the bare "document" trigger. A person who says "show me the fees
    # and rates document" is asking for a file, not for eligibility paperwork,
    # but this added "eligibility" to the required fields and the validator
    # then refused to resolve the page that held the document. Same failure
    # shape as PATCH 2: a requirement the user never stated becoming part of
    # the pass condition. "documents required"-style questions still land here
    # via "requirement".
    if "eligibility" in lower or "requirement" in lower:
        fields.append("eligibility")
    # --- END PATCH 10 ---
    # --- PATCH 2 (vendor) ---------------------------------------------------
    # Was:
    #     if not fields:
    #         if "card" in lower or "compare" in lower:
    #             fields = ["fees", "benefits"]
    #         else:
    #             fields = ["overview"]
    #
    # The `"card" in lower` half invented "fees" and "benefits" for any task
    # merely containing the word "card". The validator then required both
    # before it would resolve, so an open question like "Tell me about Banque
    # Misr payment cards" could only ever come back PARTIAL -- the page
    # answered what was asked and was rejected for not answering two things
    # nobody asked. It was also the only category vocabulary in this function.
    #
    # The `"compare" in lower` half is kept: a comparison with no stated
    # dimension genuinely needs axes to compare on, and "fees, benefits" is a
    # reasonable default there. Narrowing rather than deleting also keeps their
    # test_dynamic_plan_expansion passing on its exact expected sub-goal text.
    if not fields:
        if "compare" in lower:
            fields = ["fees", "benefits"]
        else:
            fields = ["overview"]
    return fields
    # --- END PATCH 2 ---


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
