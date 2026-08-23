"""End-to-end simulated integration scenarios across real Banque Misr fixtures."""

from person_b.adapters.fixture_loader import FixtureLoader
from person_b.planning.expansion import expand_plan
from person_b.planning.planner import (
    apply_validation,
    next_pending_sub_goal,
    plan_task,
)
from person_b.reasoning.compare import compare_items
from person_b.reasoning.synthesize import synthesize
from person_b.validation.validator import validate
from person_b.verification.self_check import finalize, validate_answer


def test_end_to_end_credit_cards_research_scenario(fixture_loader: FixtureLoader):
    """
    Simulate full Person A + Person B research lifecycle for:
    'Find all Banque Misr credit cards and compare fees and benefits'
    """
    # Step 1: User Task -> Plan
    user_task = "Find all Banque Misr credit cards and compare fees and benefits"
    plan = plan_task(user_task)
    assert len(plan.sub_goals) == 1

    # Step 2: Person A navigates & returns credit-cards-list fixture
    sub_goal_1 = next_pending_sub_goal(plan)
    list_page = fixture_loader.load_page_by_url("https://www.banquemisr.com/Home/SMEs/Retail%20Banking/Pages/Cards/Credit%20Cards%20List")

    verdict_1 = validate(
        sub_goal_1,
        list_page,
        source_url=list_page.source_url,
        content_type=list_page.content_type,
    )
    assert verdict_1["resolved"] is True
    plan = apply_validation(plan, sub_goal_1, verdict_1)

    # Step 3: Dynamic Expansion adds discovered cards
    plan = expand_plan(plan, verdict_1)
    assert len(plan.sub_goals) >= 10

    # Step 4: Person A navigates to Classic Credit Card detail page
    sub_goal_classic = next(sg for sg in plan.sub_goals if "Classic Credit Card" in sg.question)
    classic_page = fixture_loader.load_page_by_url("https://www.banquemisr.com/Home/SMEs/Retail%20Banking/Pages/Cards/Credit%20Cards%20Pages/Classic%20Credit%20Cards")

    verdict_classic = validate(
        sub_goal_classic,
        classic_page,
        source_url=classic_page.source_url,
        content_type=classic_page.content_type,
    )
    assert verdict_classic["resolved"] is True
    plan = apply_validation(plan, sub_goal_classic, verdict_classic)

    # Step 5: Person A supplies central Fee PDF
    fee_pdf = fixture_loader.load_pdf("usage-limits-and-fees-en.pdf")
    fee_pdf.source_url = "https://www.banquemisr.com/fee-schedule.pdf"

    # Step 6: Synthesis across validated results
    draft = synthesize(
        user_task=user_task,
        plan=plan,
        validated_results=[verdict_1, verdict_classic],
    )
    assert len(draft["claims"]) >= 2
    assert len(draft["sources"]) >= 2

    # Step 7: Anti-hallucination verification
    visited_pages = [list_page, classic_page, fee_pdf]
    checked = validate_answer(draft["claims"], visited_pages=visited_pages, strict=True)
    assert checked["all_supported"] is True

    # Step 8: Finalization
    final = finalize(draft, checked)
    assert len(final["source_urls"]) >= 2
    assert len(final["verified_claims"]) >= 2
    assert "Banque Misr" in final["answer"]


def test_end_to_end_unreadable_pdf_trap_scenario(fixture_loader: FixtureLoader):
    """
    Simulate trap scenario where requested document is an unreadable image-only PDF.
    Verifies that system explicitly reports not found instead of fabricating content.
    """
    user_task = "What are the text instructions to activate cards through ATM from the guide?"
    plan = plan_task(user_task)

    atm_pdf = fixture_loader.load_pdf("media-guide-to-activate-debit-and-pre-paid-cards-through-the-atm-ashx.pdf")
    atm_pdf.source_url = "https://www.banquemisr.com/atm-guide.pdf"

    verdict = validate(plan.sub_goals[0], atm_pdf, source_url=atm_pdf.source_url)
    assert verdict["resolved"] is False
    assert verdict["status"] == "unreadable"

    # Plan marks not available
    plan = apply_validation(plan, plan.sub_goals[0], verdict)

    draft = synthesize(user_task, plan=plan, validated_results=[verdict])
    checked = validate_answer(draft["claims"], visited_pages=[atm_pdf])
    final = finalize(draft, checked)

    assert "unreadable" in verdict["reason"].lower() or "not available" in str(final).lower()
