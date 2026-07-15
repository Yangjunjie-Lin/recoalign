import pytest

from recoalign.schema_validation import SchemaValidationError, validate_payload


def test_metrics_schema_rejects_boolean_values() -> None:
    with pytest.raises(SchemaValidationError):
        validate_payload("metrics", {"accuracy": True})
