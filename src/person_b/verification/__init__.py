"""Source attribution and anti-hallucination verification package."""

from person_b.verification.attribution import attribute_sources
from person_b.verification.self_check import finalize, validate_answer

__all__ = ["attribute_sources", "validate_answer", "finalize"]
