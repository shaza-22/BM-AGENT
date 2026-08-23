"""Integration tests for Person B benchmark evaluation runner and metrics."""

from person_b.evaluation.evaluator import PersonBEvaluator
from person_b.evaluation.metrics import EvaluationMetric, EvaluationResult


def test_evaluator_runs_all_benchmark_categories():
    evaluator = PersonBEvaluator()
    result = evaluator.run_suite()

    assert isinstance(result, EvaluationResult)
    assert result.passed is True
    assert result.total_cases == 8
    assert result.passed_cases == 8
    assert result.pass_rate == 1.0

    # Verify all required positive metrics are populated and >= 0.9
    for metric_name in (
        "validation_accuracy",
        "false_positive_defense_accuracy",
        "field_extraction_correctness",
        "not_found_correctness",
        "claim_support_rate",
        "source_attribution_rate",
        "dynamic_expansion_success",
    ):
        assert metric_name in result.metrics
        metric = result.metrics[metric_name]
        assert isinstance(metric, EvaluationMetric)
        assert metric.score >= 0.9, f"Metric '{metric_name}' score {metric.score} is below 0.9"

    # Verify false positive error rate is 0.0
    assert "false_positive_rate" in result.metrics
    assert result.metrics["false_positive_rate"].score == 0.0

    # Verify summary text output is clean and formatted
    summary = result.summary_text()
    assert "PERSON B EVALUATION REPORT: PASSED" in summary
    assert "validation_accuracy" in summary
