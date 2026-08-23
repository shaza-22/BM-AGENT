"""Benchmark evaluation runner against offline Banque Misr fixture scenarios."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from person_b.adapters.fixture_loader import FixtureLoader
from person_b.evaluation.metrics import EvaluationMetric, EvaluationResult
from person_b.extraction.extractor import extract_content
from person_b.models import ComparisonItem, SubGoal, ValidationStatus
from person_b.planning.expansion import expand_plan
from person_b.planning.planner import apply_validation, next_pending_sub_goal, plan_task
from person_b.reasoning.compare import compare_items
from person_b.reasoning.recommend import recommend
from person_b.reasoning.synthesize import synthesize
from person_b.validation.validator import validate
from person_b.verification.self_check import finalize, validate_answer


class PersonBEvaluator:
    """Runs 8 evaluation benchmark categories against offline fixtures and computes metrics."""

    def __init__(self, fixtures_dir: Optional[str] = None) -> None:
        self.loader = FixtureLoader(fixtures_dir)

    def run_suite(self, categories: Optional[List[str]] = None) -> EvaluationResult:
        """Execute all or requested benchmark evaluation categories."""
        cases: List[Dict[str, Any]] = []
        metrics_dict: Dict[str, EvaluationMetric] = {}

        # 1. Lookup Category
        case_lookup = self._eval_lookup()
        cases.append(case_lookup)

        # 2. Comparison Category
        case_compare = self._eval_comparison()
        cases.append(case_compare)

        # 3. Recommendation Category
        case_rec = self._eval_recommendation()
        cases.append(case_rec)

        # 4. Dynamic Multi-step Category
        case_dynamic = self._eval_dynamic_multistep()
        cases.append(case_dynamic)

        # 5. Trap / Not Available Category
        case_trap = self._eval_trap_not_available()
        cases.append(case_trap)

        # 6. Unreadable PDF Category
        case_pdf = self._eval_unreadable_pdf()
        cases.append(case_pdf)

        # 7. Misleading Title Category
        case_title = self._eval_misleading_title()
        cases.append(case_title)

        # 8. Boilerplate False Positive Defense Category
        case_boilerplate = self._eval_boilerplate_false_positive()
        cases.append(case_boilerplate)

        # Compute Aggregate Metrics
        total_cases = len(cases)
        passed_cases = sum(1 for c in cases if c["passed"])
        failures = [c for c in cases if not c["passed"]]

        # Metric: Validation Accuracy
        val_total = sum(c.get("val_total", 1) for c in cases)
        val_passed = sum(c.get("val_passed", 1 if c["passed"] else 0) for c in cases)
        metrics_dict["validation_accuracy"] = EvaluationMetric(
            name="validation_accuracy",
            score=val_passed / val_total if val_total else 1.0,
            total=val_total,
            passed=val_passed,
        )

        # Metric: False Positive Defense Accuracy & False Positive Rate
        fp_total = sum(c.get("fp_total", 0) for c in cases)
        fp_count = sum(c.get("fp_count", 0) for c in cases)
        metrics_dict["false_positive_defense_accuracy"] = EvaluationMetric(
            name="false_positive_defense_accuracy",
            score=(fp_total - fp_count) / fp_total if fp_total else 1.0,
            total=fp_total,
            passed=fp_total - fp_count,
        )
        metrics_dict["false_positive_rate"] = EvaluationMetric(
            name="false_positive_rate",
            score=fp_count / fp_total if fp_total else 0.0,
            total=fp_total,
            passed=fp_count,
        )

        # Metric: Extraction Correctness
        ext_total = sum(c.get("ext_total", 1) for c in cases)
        ext_passed = sum(c.get("ext_passed", 1 if c["passed"] else 0) for c in cases)
        metrics_dict["field_extraction_correctness"] = EvaluationMetric(
            name="field_extraction_correctness",
            score=ext_passed / ext_total if ext_total else 1.0,
            total=ext_total,
            passed=ext_passed,
        )

        # Metric: Not-Found Correctness (Traps & Unreadable handled accurately)
        nf_cases = [c for c in cases if c["category"] in ("trap_not_available", "unreadable_pdf")]
        nf_total = len(nf_cases)
        nf_passed = sum(1 for c in nf_cases if c["passed"])
        metrics_dict["not_found_correctness"] = EvaluationMetric(
            name="not_found_correctness",
            score=nf_passed / nf_total if nf_total else 1.0,
            total=nf_total,
            passed=nf_passed,
        )

        # Metric: Claim Support Rate
        cs_total = sum(c.get("claims_total", 1) for c in cases)
        cs_passed = sum(c.get("claims_supported", 1 if c["passed"] else 0) for c in cases)
        metrics_dict["claim_support_rate"] = EvaluationMetric(
            name="claim_support_rate",
            score=cs_passed / cs_total if cs_total else 1.0,
            total=cs_total,
            passed=cs_passed,
        )

        # Metric: Source Attribution Rate
        sa_total = sum(c.get("claims_total", 1) for c in cases)
        sa_passed = sum(c.get("claims_sourced", 1 if c["passed"] else 0) for c in cases)
        metrics_dict["source_attribution_rate"] = EvaluationMetric(
            name="source_attribution_rate",
            score=sa_passed / sa_total if sa_total else 1.0,
            total=sa_total,
            passed=sa_passed,
        )

        # Metric: Dynamic Expansion Success
        dyn_cases = [c for c in cases if c["category"] == "dynamic_multistep"]
        dyn_total = len(dyn_cases)
        dyn_passed = sum(1 for c in dyn_cases if c["passed"])
        metrics_dict["dynamic_expansion_success"] = EvaluationMetric(
            name="dynamic_expansion_success",
            score=dyn_passed / dyn_total if dyn_total else 1.0,
            total=dyn_total,
            passed=dyn_passed,
        )

        return EvaluationResult(
            passed=(passed_cases == total_cases),
            total_cases=total_cases,
            passed_cases=passed_cases,
            metrics=metrics_dict,
            case_results=cases,
            failures=failures,
        )

    def _eval_lookup(self) -> Dict[str, Any]:
        """Case 1: Direct lookup for Classic Credit Card issuance fee."""
        ctx = self.loader.load_text("home-smes-retail-banking-pages-cards-credit-cards-pages-classic-credit-cards.txt")
        ctx.source_url = "https://www.banquemisr.com/classic"
        sg = SubGoal(id="sg_lookup", question="Find issuance fee for Classic Credit Card", target_fields=["fees"], metadata={"target_entity": "Classic Credit Card"})

        verdict = validate(sg, ctx, source_url=ctx.source_url)
        draft = synthesize("What is Classic issuance fee?", validated_results=[verdict])
        checked = validate_answer(draft["claims"], visited_pages=[ctx], strict=True)
        final = finalize(draft, checked)

        passed = (verdict["resolved"] is True and checked["all_supported"] is True and "250" in str(verdict["extracted"]))
        return {
            "category": "lookup",
            "name": "Classic Credit Card Fee Lookup",
            "passed": passed,
            "reason": "OK" if passed else "Failed fee lookup or verification",
            "claims_total": len(draft["claims"]),
            "claims_supported": len(checked["passed"]),
            "claims_sourced": len([c for c in draft["claims"] if c.get("source_url")]),
        }

    def _eval_comparison(self) -> Dict[str, Any]:
        """Case 2: Structured product evidence comparison."""
        it1 = ComparisonItem(name="Classic Credit Card", attributes={"issuance_fee": "250 EGP", "interest_rate": "4% monthly"})
        it2 = ComparisonItem(name="Gold Credit Card", attributes={"issuance_fee": "500 EGP", "interest_rate": "4% monthly"})

        res = compare_items([it1, it2])
        passed = ("differences" in res and res["differences"]["issuance_fee"]["Classic Credit Card"] == "250 EGP")
        return {
            "category": "comparison",
            "name": "Card Fee Comparison",
            "passed": passed,
            "reason": "OK" if passed else "Comparison difference matrix mismatch",
        }

    def _eval_recommendation(self) -> Dict[str, Any]:
        """Case 3: Recommendation based strictly on stated criteria."""
        it1 = ComparisonItem(name="Classic Credit Card", attributes={"issuance_fee": "250 EGP"})
        it2 = ComparisonItem(name="Platinum Card", attributes={"issuance_fee": "1000 EGP", "travel": "Free lounge access"})

        rec_fee = recommend({"priority": "lowest fee"}, [it1, it2])
        rec_travel = recommend({"priority": "international travel"}, [it1, it2])

        passed = (rec_fee["recommended_item"] == "Classic Credit Card" and rec_travel["recommended_item"] == "Platinum Card")
        return {
            "category": "recommendation",
            "name": "Criteria-Driven Recommendation",
            "passed": passed,
            "reason": "OK" if passed else "Recommendation criteria matching failed",
        }

    def _eval_dynamic_multistep(self) -> Dict[str, Any]:
        """Case 4: Discovery to dynamic plan expansion."""
        plan = plan_task("Find all Banque Misr credit cards and compare fees and benefits")
        list_ctx = self.loader.load_text("home-smes-retail-banking-pages-cards-credit-cards-list.txt")
        verdict = validate(plan.sub_goals[0], list_ctx)

        expanded = expand_plan(plan, verdict)
        passed = (verdict["resolved"] is True and len(expanded.sub_goals) >= 10)
        return {
            "category": "dynamic_multistep",
            "name": "Card List Discovery and Expansion",
            "passed": passed,
            "reason": "OK" if passed else f"Expansion produced {len(expanded.sub_goals)} subgoals",
        }

    def _eval_trap_not_available(self) -> Dict[str, Any]:
        """Case 5: Trap question for unsupported/non-existent feature."""
        task = "Find Banque Misr cryptocurrency trading margin account fees"
        plan = plan_task(task)
        classic_ctx = self.loader.load_text("home-smes-retail-banking-pages-cards-credit-cards-pages-classic-credit-cards.txt")

        verdict = validate(plan.sub_goals[0], classic_ctx)
        # Should NOT resolve cryptocurrency on a credit card page
        passed = (verdict["resolved"] is False)
        return {
            "category": "trap_not_available",
            "name": "Cryptocurrency Margin Trading Trap",
            "passed": passed,
            "reason": "OK" if passed else "False resolution on unsupported trap query",
            "fp_total": 1,
            "fp_count": 0 if passed else 1,
        }

    def _eval_unreadable_pdf(self) -> Dict[str, Any]:
        """Case 6: Image-only ATM activation guide PDF."""
        pdf_ctx = self.loader.load_pdf("media-guide-to-activate-debit-and-pre-paid-cards-through-the-atm-ashx.pdf")
        sg = SubGoal(id="sg_pdf", question="Find ATM card activation steps")

        verdict = validate(sg, pdf_ctx)
        passed = (verdict["resolved"] is False and verdict["status"] == ValidationStatus.UNREADABLE.value)
        return {
            "category": "unreadable_pdf",
            "name": "Image-Only ATM Guide PDF",
            "passed": passed,
            "reason": "OK" if passed else "Failed unreadable PDF handling",
            "fp_total": 1,
            "fp_count": 0 if passed else 1,
        }

    def _eval_misleading_title(self) -> Dict[str, Any]:
        """Case 7: Accounts & Deposits fixture with misleading first-line Classic Credit Card title."""
        ctx = self.loader.load_text("home-smes-retail-banking-accounts-and-deposits.txt")

        # Must NOT resolve Classic Card Fees
        sg_card = SubGoal(id="sg_card", question="Find Classic Credit Card fees", target_fields=["fees"], metadata={"target_entity": "Classic Credit Card"})
        verdict_card = validate(sg_card, ctx)

        # MUST resolve Accounts & Deposits
        sg_acc = SubGoal(id="sg_acc", question="Find Banque Misr accounts and deposits", target_fields=["accounts_list"])
        verdict_acc = validate(sg_acc, ctx)

        passed = (verdict_card["resolved"] is False and verdict_acc["resolved"] is True)
        return {
            "category": "misleading_title",
            "name": "Accounts & Deposits Misleading Title Defense",
            "passed": passed,
            "reason": "OK" if passed else "Failed misleading title disambiguation",
            "fp_total": 1,
            "fp_count": 0 if verdict_card["resolved"] is False else 1,
        }

    def _eval_boilerplate_false_positive(self) -> Dict[str, Any]:
        """Case 8: Keyword only in navigation/footer boilerplate must not resolve sub-goal."""
        ctx = self.loader.load_text("home-smes-retail-banking-consumer-loans.txt")
        # 'Board Members' or 'Compliance' appears in nav boilerplate
        sg = SubGoal(id="sg_board", question="Find Board Members biographies and details", target_fields=["board_biographies"], metadata={"target_entity": "Board Members"})

        verdict = validate(sg, ctx)
        passed = (verdict["resolved"] is False)
        return {
            "category": "boilerplate_false_positive",
            "name": "Navigation Boilerplate False-Positive Defense",
            "passed": passed,
            "reason": "OK" if passed else "False resolution from nav boilerplate",
            "fp_total": 1,
            "fp_count": 0 if passed else 1,
        }
