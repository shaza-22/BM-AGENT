"""Public API facade for Person B.

This module provides the primary entry points for Person A integration.
"""

from person_b.adapters.fixture_loader import FixtureLoader
from person_b.adapters.reasoning_backend import (
    FakeReasoningBackend,
    ReasoningBackend,
    get_reasoning_backend,
)
from person_b.config import PersonBConfig, default_config
from person_b.errors import (
    AdapterError,
    ConfigurationError,
    ExtractionError,
    FixtureError,
    PersonBError,
    PlanningError,
    ReasoningError,
    ValidationError,
    VerificationError,
)
from person_b.evaluation.evaluator import PersonBEvaluator
from person_b.evaluation.metrics import EvaluationMetric, EvaluationResult
from person_b.extraction.extractor import extract_content
from person_b.logging.jsonl_logger import PersonBLogEvent, PersonBLogger
from person_b.models import (
    Claim,
    ClaimCheck,
    ClaimStatus,
    ComparisonItem,
    ComparisonResult,
    ContentType,
    Evidence,
    EvidenceLocation,
    ExtractedField,
    ExtractionLocationType,
    ExtractionResult,
    FinalAnswer,
    Goal,
    PageContext,
    Plan,
    SubGoal,
    SubGoalStatus,
    SynthesisResult,
    TaskType,
    ValidationResult,
    ValidationStatus,
)
from person_b.planning.expansion import expand_plan
from person_b.planning.planner import (
    apply_validation,
    mark_not_available,
    next_pending_sub_goal,
    plan_task,
)
from person_b.reasoning.compare import compare_items
from person_b.reasoning.recommend import recommend
from person_b.reasoning.synthesize import synthesize
from person_b.validation.validator import validate
from person_b.verification.attribution import attribute_sources
from person_b.verification.self_check import finalize, validate_answer

__all__ = [
    # Primary Integration Functions
    "validate",
    "plan_task",
    "next_pending_sub_goal",
    "apply_validation",
    "mark_not_available",
    "expand_plan",
    "extract_content",
    "compare_items",
    "recommend",
    "synthesize",
    "attribute_sources",
    "validate_answer",
    "finalize",
    # Config & Adapters
    "PersonBConfig",
    "default_config",
    "FixtureLoader",
    "ReasoningBackend",
    "FakeReasoningBackend",
    "get_reasoning_backend",
    # Logging & Evaluation
    "PersonBLogger",
    "PersonBLogEvent",
    "PersonBEvaluator",
    "EvaluationMetric",
    "EvaluationResult",
    # Core Models
    "Goal",
    "SubGoal",
    "Plan",
    "PageContext",
    "Evidence",
    "EvidenceLocation",
    "ExtractedField",
    "ExtractionResult",
    "ValidationResult",
    "ComparisonItem",
    "ComparisonResult",
    "Claim",
    "ClaimCheck",
    "SynthesisResult",
    "FinalAnswer",
    # Enums
    "SubGoalStatus",
    "ValidationStatus",
    "ContentType",
    "ExtractionLocationType",
    "ClaimStatus",
    "TaskType",
    # Errors
    "PersonBError",
    "ConfigurationError",
    "PlanningError",
    "ExtractionError",
    "ValidationError",
    "ReasoningError",
    "VerificationError",
    "AdapterError",
    "FixtureError",
]
