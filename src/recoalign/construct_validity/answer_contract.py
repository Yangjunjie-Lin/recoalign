"""Frozen primary and secondary answer contracts for PIVOT_EXP_A3."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

ANSWER_CONTRACT_VERSION = "answer-contract-v2"
REGISTERED_CHOICE_IDS = ("1", "2", "3", "4")
PRIMARY_COMPLETION_PREFIX = "FINAL_CHOICE="
SECONDARY_GRAMMAR = r"^\s*FINAL_CHOICE\s*=\s*([1-4])\s*$"
_SECONDARY_PATTERN = re.compile(SECONDARY_GRAMMAR, flags=re.ASCII)


@dataclass(frozen=True)
class ParsedChoice:
    raw_output: str
    valid: bool
    choice_id: str | None
    failure_reason: str | None
    contract_version: str = ANSWER_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_output": self.raw_output,
            "valid": self.valid,
            "choice_id": self.choice_id,
            "failure_reason": self.failure_reason,
            "contract_version": self.contract_version,
        }


def parse_final_choice(raw_output: str) -> ParsedChoice:
    """Parse only the preregistered ASCII, uppercase, one-field grammar.

    The function deliberately performs no Unicode normalization, case folding, synonym lookup,
    alias matching, substring search, or reasoning-trace extraction.
    """

    raw = str(raw_output)
    match = _SECONDARY_PATTERN.fullmatch(raw)
    if match is not None:
        return ParsedChoice(
            raw_output=raw,
            valid=True,
            choice_id=match.group(1),
            failure_reason=None,
        )
    return ParsedChoice(
        raw_output=raw,
        valid=False,
        choice_id=None,
        failure_reason=_failure_reason(raw),
    )


def parser_contract_cases() -> tuple[dict[str, Any], ...]:
    """Return the frozen adversarial validation inventory and expected result."""

    return (
        {"id": "exact_valid", "raw": "FINAL_CHOICE=1", "valid": True, "choice": "1"},
        {
            "id": "leading_trailing_whitespace",
            "raw": " \tFINAL_CHOICE = 2\r\n",
            "valid": True,
            "choice": "2",
        },
        {
            "id": "uppercase_contract",
            "raw": "FINAL_CHOICE=3",
            "valid": True,
            "choice": "3",
        },
        {
            "id": "lowercase_rejected",
            "raw": "final_choice=3",
            "valid": False,
            "choice": None,
        },
        {
            "id": "full_sentence_rejected",
            "raw": "The answer is FINAL_CHOICE=1.",
            "valid": False,
            "choice": None,
        },
        {
            "id": "multiple_option_rejected",
            "raw": "FINAL_CHOICE=1 or FINAL_CHOICE=2",
            "valid": False,
            "choice": None,
        },
        {
            "id": "invalid_option_rejected",
            "raw": "FINAL_CHOICE=5",
            "valid": False,
            "choice": None,
        },
        {
            "id": "alias_collision_rejected",
            "raw": "FINAL_CHOICE=E1",
            "valid": False,
            "choice": None,
        },
        {
            "id": "reasoning_trace_rejected",
            "raw": "Because E2 is blue, FINAL_CHOICE=2",
            "valid": False,
            "choice": None,
        },
        {
            "id": "truncated_output_rejected",
            "raw": "FINAL_CHOICE=",
            "valid": False,
            "choice": None,
        },
        {
            "id": "unicode_equals_rejected",
            "raw": "FINAL_CHOICE＝1",
            "valid": False,
            "choice": None,
        },
        {
            "id": "unicode_digit_rejected",
            "raw": "FINAL_CHOICE=１",
            "valid": False,
            "choice": None,
        },
        {
            "id": "duplicate_final_field_rejected",
            "raw": "FINAL_CHOICE=1\nFINAL_CHOICE=1",
            "valid": False,
            "choice": None,
        },
        {
            "id": "option_substring_false_positive_rejected",
            "raw": "FINAL_CHOICE=14",
            "valid": False,
            "choice": None,
        },
        {
            "id": "entity_alias_only_rejected",
            "raw": "E2",
            "valid": False,
            "choice": None,
        },
    )


def validate_parser_contract() -> dict[str, Any]:
    cases = []
    for registered in parser_contract_cases():
        parsed = parse_final_choice(str(registered["raw"]))
        passed = (
            parsed.valid is bool(registered["valid"])
            and parsed.choice_id == registered["choice"]
        )
        cases.append(
            {
                "id": registered["id"],
                "expected_valid": bool(registered["valid"]),
                "expected_choice": registered["choice"],
                "observed": parsed.to_dict(),
                "passed": passed,
            }
        )
    return {
        "contract_version": ANSWER_CONTRACT_VERSION,
        "grammar": SECONDARY_GRAMMAR,
        "case_sensitive": True,
        "unicode_normalization": False,
        "reasoning_trace_policy": "reject_entire_output",
        "passed": all(bool(case["passed"]) for case in cases),
        "cases": cases,
    }


def _failure_reason(raw: str) -> str:
    if not raw.strip():
        return "empty"
    occurrences = len(re.findall(r"FINAL_CHOICE", raw, flags=re.ASCII))
    if occurrences > 1:
        return "duplicate_final_field"
    if not raw.lstrip().startswith("FINAL_CHOICE"):
        return "unexpected_prefix_or_reasoning_trace"
    if not re.match(r"^\s*FINAL_CHOICE\s*=", raw, flags=re.ASCII):
        return "invalid_ascii_delimiter"
    candidate = re.match(r"^\s*FINAL_CHOICE\s*=\s*(\S*)", raw, flags=re.ASCII)
    value = candidate.group(1) if candidate is not None else ""
    if not value:
        return "truncated_missing_option"
    if value[:1] not in REGISTERED_CHOICE_IDS:
        return "invalid_option"
    return "trailing_or_multiple_content"


__all__ = [
    "ANSWER_CONTRACT_VERSION",
    "PRIMARY_COMPLETION_PREFIX",
    "ParsedChoice",
    "REGISTERED_CHOICE_IDS",
    "SECONDARY_GRAMMAR",
    "parse_final_choice",
    "parser_contract_cases",
    "validate_parser_contract",
]
