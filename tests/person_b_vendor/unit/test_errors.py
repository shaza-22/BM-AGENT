"""Unit tests for Person B typed error hierarchy."""

import pytest
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


def test_error_hierarchy():
    assert issubclass(ConfigurationError, PersonBError)
    assert issubclass(PlanningError, PersonBError)
    assert issubclass(ExtractionError, PersonBError)
    assert issubclass(ValidationError, PersonBError)
    assert issubclass(ReasoningError, PersonBError)
    assert issubclass(VerificationError, PersonBError)
    assert issubclass(AdapterError, PersonBError)
    assert issubclass(FixtureError, PersonBError)


def test_error_dict_serialization():
    err = ExtractionError("Malformed pipe table row", {"line": 15, "raw": "bad | row"})
    d = err.to_dict()
    assert d["error_type"] == "ExtractionError"
    assert d["message"] == "Malformed pipe table row"
    assert d["details"]["line"] == 15
    assert "details:" in str(err)
