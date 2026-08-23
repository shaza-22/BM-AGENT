"""Verify all Person B public symbols, submodules, and models import cleanly."""

import person_b
from person_b import (
    # Facade functions
    apply_validation,
    attribute_sources,
    compare_items,
    expand_plan,
    extract_content,
    finalize,
    mark_not_available,
    next_pending_sub_goal,
    plan_task,
    recommend,
    synthesize,
    validate,
    validate_answer,
    # Config & Adapters
    FakeReasoningBackend,
    FixtureLoader,
    PersonBConfig,
    ReasoningBackend,
    get_reasoning_backend,
    # Core Models
    Claim,
    ClaimCheck,
    ComparisonItem,
    ComparisonResult,
    Evidence,
    EvidenceLocation,
    ExtractedField,
    ExtractionResult,
    FinalAnswer,
    Goal,
    PageContext,
    Plan,
    SubGoal,
    SynthesisResult,
    ValidationResult,
    # Enums
    ClaimStatus,
    ContentType,
    ExtractionLocationType,
    SubGoalStatus,
    TaskType,
    ValidationStatus,
    # Errors
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


def test_top_level_exports():
    """Verify that key public functions are callable and exported."""
    assert callable(validate)
    assert callable(plan_task)
    assert callable(next_pending_sub_goal)
    assert callable(expand_plan)
    assert callable(apply_validation)
    assert callable(mark_not_available)
    assert callable(extract_content)
    assert callable(compare_items)
    assert callable(synthesize)
    assert callable(validate_answer)
    assert callable(finalize)


def test_package_version():
    """Verify package version is defined."""
    assert hasattr(person_b, "__version__")
    assert isinstance(person_b.__version__, str)
