"""Unit tests for Person B data models and serialization."""

import json
from person_b.models import (
    Claim,
    ClaimCheck,
    ClaimStatus,
    ComparisonItem,
    ComparisonResult,
    ContentType,
    Evidence,
    EvidenceLocation,
    ExtractionLocationType,
    ExtractionResult,
    FinalAnswer,
    Goal,
    PageContext,
    Plan,
    SubGoal,
    SubGoalStatus,
    SynthesisResult,
    ValidationResult,
    ValidationStatus,
)


def test_goal_serialization():
    goal = Goal(
        id="g_123",
        statement="Find credit card fees",
        original_task="What are the fees for Classic card?",
        task_type="lookup",
        constraints=["fee must be EGP"],
    )
    d = goal.to_dict()
    assert d["id"] == "g_123"
    assert d["statement"] == "Find credit card fees"
    assert d["constraints"] == ["fee must be EGP"]

    # Roundtrip from dict
    restored = Goal.from_dict(d)
    assert restored.id == goal.id
    assert restored.statement == goal.statement
    assert restored.constraints == goal.constraints


def test_subgoal_lifecycle_and_serialization():
    sg = SubGoal(
        id="sg_001",
        question="Find issuance fee for Classic card",
        status=SubGoalStatus.PENDING,
        target_fields=["issuance_fee"],
    )
    d = sg.to_dict()
    assert d["id"] == "sg_001"
    assert d["status"] == "pending"
    assert d["target_fields"] == ["issuance_fee"]

    restored = SubGoal.from_dict(d)
    assert restored.id == "sg_001"
    assert restored.status == SubGoalStatus.PENDING


def test_plan_and_subgoals():
    goal = Goal(id="g_1", statement="Research cards", original_task="Research cards")
    sg1 = SubGoal(id="sg_1", question="List cards", status=SubGoalStatus.RESOLVED)
    sg2 = SubGoal(id="sg_2", question="Get fees", status=SubGoalStatus.PENDING)
    plan = Plan(id="p_1", goal=goal, sub_goals=[sg1, sg2])

    assert len(plan.sub_goals) == 2
    assert len(plan.pending_sub_goals) == 1
    assert plan.pending_sub_goals[0].id == "sg_2"
    assert len(plan.resolved_sub_goals) == 1
    assert plan.get_sub_goal("sg_1") == sg1

    d = plan.to_dict()
    # Ensure JSON serializable
    json_str = json.dumps(d)
    assert "p_1" in json_str

    restored = Plan.from_dict(d)
    assert restored.id == "p_1"
    assert len(restored.sub_goals) == 2


def test_evidence_and_location():
    loc = EvidenceLocation(
        type=ExtractionLocationType.TEXT_TABLE,
        table_name="Fees and charges",
        row_index=2,
        col_name="Issuance",
    )
    ev = Evidence(
        field="issuance_fee",
        value="250 EGP",
        source_url="https://www.banquemisr.com/classic",
        evidence_text="Issuance fee | 250 EGP",
        location=loc,
        confidence=0.95,
    )
    d = ev.to_dict()
    assert d["field"] == "issuance_fee"
    assert d["location"]["type"] == "text_table"
    assert d["location"]["table_name"] == "Fees and charges"

    # JSON serializability
    json.dumps(d)

    restored = Evidence.from_dict(d)
    assert restored.field == "issuance_fee"
    assert restored.value == "250 EGP"
    assert restored.location.table_name == "Fees and charges"


def test_validation_result_contract_dict():
    vr = ValidationResult(
        resolved=False,
        status=ValidationStatus.UNRESOLVED,
        extracted={},
        reason="Page does not contain requested card fee table.",
        source_url="https://www.banquemisr.com/cards",
    )
    contract_d = vr.to_contract_dict()
    assert set(contract_d.keys()) == {"resolved", "extracted", "reason"}
    assert contract_d["resolved"] is False
    assert contract_d["extracted"] == {}

    full_d = vr.to_dict()
    assert full_d["resolved"] is False
    assert full_d["status"] == "unresolved"
    assert full_d["source_url"] == "https://www.banquemisr.com/cards"


def test_claim_and_claim_check():
    c = Claim(
        id="c_1",
        statement="Classic card issuance fee is 250 EGP",
        entity="Classic Credit Card",
        field="issuance_fee",
        value="250 EGP",
        source_url="https://www.banquemisr.com/classic",
    )
    chk = ClaimCheck(
        claim=c,
        status=ClaimStatus.SUPPORTED,
        reason="Verified against Fees table.",
        verified_source_url="https://www.banquemisr.com/classic",
    )
    d = chk.to_dict()
    assert d["status"] == "supported"
    assert d["claim"]["id"] == "c_1"
    json.dumps(d)
