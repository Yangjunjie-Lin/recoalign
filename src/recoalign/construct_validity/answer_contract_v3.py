"""Prospectively frozen continuation contract for PIVOT_EXP_A3R."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

CONTRACT_VERSION = "answer-contract-v3-continuation"
PROMPT_COMPLETION_PREFIX = "FINAL_CHOICE="
REGISTERED_CHOICE_IDS = ("1", "2", "3", "4")
RAW_CONTINUATION_GRAMMAR = r"^\s*([1-4])\s*$"
RECONSTRUCTED_CONTRACT_GRAMMAR = r"^\s*FINAL_CHOICE\s*=\s*([1-4])\s*$"

_RAW_PATTERN = re.compile(RAW_CONTINUATION_GRAMMAR, flags=re.ASCII)
_RECONSTRUCTED_PATTERN = re.compile(RECONSTRUCTED_CONTRACT_GRAMMAR, flags=re.ASCII)


@dataclass(frozen=True)
class ParsedContinuation:
    raw_continuation: str
    prompt_completion_prefix: str
    reconstructed_contract_output: str
    raw_valid: bool
    reconstructed_valid: bool
    choice_ids_match: bool
    choice_id: str | None
    failure_reason: str | None
    contract_version: str = CONTRACT_VERSION

    @property
    def valid(self) -> bool:
        return self.raw_valid and self.reconstructed_valid and self.choice_ids_match

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "prompt_completion_prefix": self.prompt_completion_prefix,
            "raw_continuation": self.raw_continuation,
            "reconstructed_contract_output": self.reconstructed_contract_output,
            "raw_valid": self.raw_valid,
            "reconstructed_valid": self.reconstructed_valid,
            "choice_ids_match": self.choice_ids_match,
            "valid": self.valid,
            "choice_id": self.choice_id,
            "failure_reason": self.failure_reason,
        }


def parse_continuation(raw_continuation: str) -> ParsedContinuation:
    """Validate raw continuation and its mechanical frozen-prefix reconstruction.

    The input is retained verbatim. No Unicode normalization, case folding, conversion, alias
    matching, substring search, or reasoning-trace extraction occurs.
    """

    raw = str(raw_continuation)
    reconstructed = PROMPT_COMPLETION_PREFIX + raw
    raw_match = _RAW_PATTERN.fullmatch(raw)
    reconstructed_match = _RECONSTRUCTED_PATTERN.fullmatch(reconstructed)
    raw_choice = raw_match.group(1) if raw_match is not None else None
    reconstructed_choice = (
        reconstructed_match.group(1) if reconstructed_match is not None else None
    )
    choices_match = bool(
        raw_choice is not None
        and reconstructed_choice is not None
        and raw_choice == reconstructed_choice
    )
    valid = raw_match is not None and reconstructed_match is not None and choices_match
    return ParsedContinuation(
        raw_continuation=raw,
        prompt_completion_prefix=PROMPT_COMPLETION_PREFIX,
        reconstructed_contract_output=reconstructed,
        raw_valid=raw_match is not None,
        reconstructed_valid=reconstructed_match is not None,
        choice_ids_match=choices_match,
        choice_id=raw_choice if valid else None,
        failure_reason=None if valid else _failure_reason(raw),
    )


def parser_contract_cases() -> tuple[dict[str, Any], ...]:
    valid = ("1", "2", "3", "4", " 2 ", "2\n")
    invalid = (
        "FINAL_CHOICE=2",
        "Option 2",
        "The answer is 2",
        "2 or 3",
        "22",
        "E2",
        "２",
        "two",
        "",
        "reasoning trace + 2",
    )
    return tuple(
        {"id": f"valid_{index}", "raw": raw, "valid": True}
        for index, raw in enumerate(valid, start=1)
    ) + tuple(
        {"id": f"invalid_{index}", "raw": raw, "valid": False}
        for index, raw in enumerate(invalid, start=1)
    )


def validate_parser_contract() -> dict[str, Any]:
    cases = []
    for registered in parser_contract_cases():
        parsed = parse_continuation(str(registered["raw"]))
        passed = parsed.valid is bool(registered["valid"])
        cases.append(
            {
                "id": registered["id"],
                "expected_valid": bool(registered["valid"]),
                "observed": parsed.to_dict(),
                "passed": passed,
            }
        )
    reconstructed_cases = []
    for choice_id in REGISTERED_CHOICE_IDS:
        output = PROMPT_COMPLETION_PREFIX + choice_id
        match = _RECONSTRUCTED_PATTERN.fullmatch(output)
        reconstructed_cases.append(
            {
                "output": output,
                "expected_choice_id": choice_id,
                "observed_choice_id": match.group(1) if match is not None else None,
                "passed": match is not None and match.group(1) == choice_id,
            }
        )
    return {
        "contract_version": CONTRACT_VERSION,
        "prompt_completion_prefix": PROMPT_COMPLETION_PREFIX,
        "raw_continuation_grammar": RAW_CONTINUATION_GRAMMAR,
        "reconstructed_contract_grammar": RECONSTRUCTED_CONTRACT_GRAMMAR,
        "unicode_normalization": False,
        "word_number_conversion": False,
        "substring_matching": False,
        "alias_matching": False,
        "multiple_option_acceptance": False,
        "reasoning_trace_extraction": False,
        "cases": cases,
        "reconstructed_cases": reconstructed_cases,
        "passed": all(bool(case["passed"]) for case in cases + reconstructed_cases),
    }


def _failure_reason(raw: str) -> str:
    if not raw:
        return "empty"
    if any(ord(character) > 127 for character in raw):
        return "non_ascii"
    if re.search(r"[1-4].*[1-4]", raw, flags=re.ASCII):
        return "multiple_or_multidigit_option"
    if raw.strip() in REGISTERED_CHOICE_IDS:
        return "reconstruction_mismatch"
    return "unexpected_content"


__all__ = [
    "CONTRACT_VERSION",
    "PROMPT_COMPLETION_PREFIX",
    "RAW_CONTINUATION_GRAMMAR",
    "RECONSTRUCTED_CONTRACT_GRAMMAR",
    "REGISTERED_CHOICE_IDS",
    "ParsedContinuation",
    "parse_continuation",
    "parser_contract_cases",
    "validate_parser_contract",
]
