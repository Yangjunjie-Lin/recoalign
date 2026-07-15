"""Runtime validation for committed research record contracts."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


class SchemaValidationError(ValueError):
    """Raised when a research record violates a committed JSON Schema."""


@cache
def load_schema(name: str) -> dict[str, Any]:
    """Load a JSON Schema from the repository-level ``schemas`` directory."""
    schema_path = repository_root() / "schemas" / f"{name}.schema.json"
    if not schema_path.is_file():
        raise FileNotFoundError(f"schema does not exist: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SchemaValidationError(f"schema root must be an object: {schema_path}")
    Draft202012Validator.check_schema(payload)
    return payload


def validate_payload(name: str, payload: Any) -> None:
    """Validate a payload and raise one compact, deterministic error message."""
    validator = Draft202012Validator(load_schema(name), format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.absolute_path))
    if not errors:
        return
    first = errors[0]
    location = ".".join(str(part) for part in first.absolute_path) or "<root>"
    raise SchemaValidationError(f"{name} schema violation at {location}: {first.message}")


def repository_root() -> Path:
    """Return the repository root for editable research installations."""
    root = Path(__file__).resolve().parents[2]
    if not (root / "schemas").is_dir():
        raise FileNotFoundError(
            "ReCoAlign schemas were not found. Install the project in editable mode from the "
            "repository root."
        )
    return root
