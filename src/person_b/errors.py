"""Typed exception hierarchy for Person B."""

from typing import Any, Dict, Optional


class PersonBError(Exception):
    """Base exception for all Person B errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


class ConfigurationError(PersonBError):
    """Raised when configuration is invalid or missing required parameters."""


class PlanningError(PersonBError):
    """Raised when task planning or sub-goal generation fails."""


class ExtractionError(PersonBError):
    """Raised when extracting content, text tables, or PDF tables fails."""


class ValidationError(PersonBError):
    """Raised when sub-goal validation encounters an unrecoverable internal error."""


class ReasoningError(PersonBError):
    """Raised when reasoning, item comparison, or synthesis fails."""


class VerificationError(PersonBError):
    """Raised when claim verification or self-check encounters an unrecoverable failure."""


class AdapterError(PersonBError):
    """Raised when an external adapter (e.g. LLM provider) fails."""


class FixtureError(PersonBError):
    """Raised when loading or parsing development/test fixtures fails."""
