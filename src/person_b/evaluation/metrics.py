"""Evaluation metrics definitions and aggregators for Person B."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class EvaluationMetric:
    """Single evaluation metric score with total and passed tallies."""
    name: str
    score: float
    total: int = 1
    passed: int = 1
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "total": self.total,
            "passed": self.passed,
            "details": self.details,
        }


@dataclass
class EvaluationResult:
    """Overall evaluation benchmark run results across all categories."""
    passed: bool
    total_cases: int
    passed_cases: int
    metrics: Dict[str, EvaluationMetric] = field(default_factory=dict)
    case_results: List[Dict[str, Any]] = field(default_factory=list)
    failures: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return (self.passed_cases / self.total_cases) if self.total_cases > 0 else 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "pass_rate": round(self.pass_rate, 4),
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "case_results": self.case_results,
            "failures": self.failures,
        }

    def summary_text(self) -> str:
        lines = [
            "=" * 60,
            f"PERSON B EVALUATION REPORT: {'PASSED' if self.passed else 'FAILED'}",
            f"Overall: {self.passed_cases}/{self.total_cases} cases passed ({self.pass_rate * 100:.1f}%)",
            "=" * 60,
            "METRICS:",
        ]
        for name, m in self.metrics.items():
            lines.append(f"  - {name:<32}: {m.score * 100:>6.1f}% ({m.passed}/{m.total})")
        if self.failures:
            lines.append("-" * 60)
            lines.append("FAILURES:")
            for f in self.failures:
                lines.append(f"  * [{f.get('category')}] {f.get('name')}: {f.get('reason')}")
        lines.append("=" * 60)
        return "\n".join(lines)
