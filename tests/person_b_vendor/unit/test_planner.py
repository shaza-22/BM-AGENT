"""Unit tests for task understanding, initial planning, and dynamic expansion."""

from person_b.config import PersonBConfig
from person_b.models import (
    ExtractionResult,
    SubGoalStatus,
    TaskType,
    ValidationResult,
)
from person_b.planning.expansion import expand_plan
from person_b.planning.planner import (
    apply_validation,
    mark_not_available,
    next_pending_sub_goal,
    plan_task,
)


def test_planner_direct_lookup():
    task = "What is the issuance fee for Banque Misr Classic Credit Card?"
    plan = plan_task(task)

    assert plan.goal.statement == task
    assert plan.goal.task_type == TaskType.LOOKUP.value
    assert len(plan.sub_goals) == 1
    assert "fees" in plan.goal.metadata.get("target_fields", [])
    assert plan.sub_goals[0].status == SubGoalStatus.PENDING


def test_planner_comparison():
    task = "Compare all Banque Misr credit cards by fees and benefits."
    plan = plan_task(task)

    assert plan.goal.task_type == TaskType.COMPARISON.value
    assert len(plan.sub_goals) == 1
    assert plan.sub_goals[0].question == "Find the list of Banque Misr credit cards"
    assert plan.sub_goals[0].target_fields == ["credit_cards_list"]


def test_planner_recommendation():
    task = "Which credit card is best for international travel?"
    plan = plan_task(task)

    assert plan.goal.task_type == TaskType.RECOMMENDATION.value
    assert len(plan.sub_goals) == 1


def test_planner_consumer_loans():
    task = "Find consumer loans and interest terms"
    plan = plan_task(task)

    assert len(plan.sub_goals) == 1
    assert "loan" in plan.sub_goals[0].question.lower()


def test_planner_no_hardcoded_card_names():
    """Verify initial plan does not hardcode card names before site reveals them."""
    plan = plan_task("Find all credit cards and compare them")
    sub_goal_texts = " ".join(sg.question for sg in plan.sub_goals)
    for card in ("Classic", "Gold", "Titanium", "Platinum", "World Elite", "Visa Infinite"):
        assert card not in sub_goal_texts, f"Initial plan must not hardcode card '{card}'"


def test_dynamic_plan_expansion():
    plan = plan_task("Compare all Banque Misr credit cards")
    extraction = ExtractionResult(
        status="success",
        extracted={
            "entities": [
                {"name": "Classic Credit Card"},
                {"name": "Gold Credit Card"},
                {"name": "Titanium Credit Card"},
            ]
        },
    )

    expanded = expand_plan(plan, extraction)
    assert len(expanded.sub_goals) == 4  # 1 initial + 3 discovered
    assert expanded.sub_goals[1].question == "Find fees, benefits for Classic Credit Card"
    assert expanded.sub_goals[1].parent_id == "sg_001"
    assert expanded.sub_goals[2].question == "Find fees, benefits for Gold Credit Card"
    assert expanded.sub_goals[3].question == "Find fees, benefits for Titanium Credit Card"
    assert expanded.version == 2


def test_dynamic_expansion_deduplication():
    plan = plan_task("Compare all Banque Misr credit cards")
    extraction = ExtractionResult(
        status="success",
        extracted={
            "entities": [
                {"name": "Classic Credit Card"},
                {"name": "Gold Credit Card"},
            ]
        },
    )

    expanded = expand_plan(plan, extraction)
    assert len(expanded.sub_goals) == 3

    # Re-expand with identical payload
    re_expanded = expand_plan(expanded, extraction)
    assert len(re_expanded.sub_goals) == 3, "Duplicate expansion must be prevented"


def test_dynamic_expansion_cap():
    cfg = PersonBConfig(max_expansion_sub_goals=5)
    plan = plan_task("Compare all cards", config=cfg)
    extraction = ExtractionResult(
        status="success",
        extracted={
            "entities": [{"name": f"Card {i}"} for i in range(20)]
        },
    )

    expanded = expand_plan(plan, extraction, config=cfg)
    assert len(expanded.sub_goals) <= 5, "Expansion must not exceed max_expansion_sub_goals cap"


def test_plan_lifecycle_and_helpers():
    plan = plan_task("Research loans")
    sg = next_pending_sub_goal(plan)
    assert sg is not None
    assert sg.id == "sg_001"

    # Apply resolved validation
    apply_validation(plan, sg, {"resolved": True})
    assert sg.status == SubGoalStatus.RESOLVED
    assert next_pending_sub_goal(plan) is None

    # Mark not available
    plan2 = plan_task("Research unsupported feature")
    sg2 = next_pending_sub_goal(plan2)
    mark_not_available(plan2, sg2, reason="Page limits exhausted")
    assert sg2.status == SubGoalStatus.NOT_AVAILABLE
    assert sg2.metadata["not_available_reason"] == "Page limits exhausted"
