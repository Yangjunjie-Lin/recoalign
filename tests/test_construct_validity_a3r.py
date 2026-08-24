from __future__ import annotations

import gzip
import json
from pathlib import Path

from recoalign.construct_validity.a3r import (
    _amend_trial,
    _read_development_records,
    _required_development_smoke_rows,
    load_config,
)
from recoalign.construct_validity.answer_contract import parse_final_choice
from recoalign.construct_validity.answer_contract_v3 import (
    CONTRACT_VERSION,
    PROMPT_COMPLETION_PREFIX,
    parse_continuation,
    validate_parser_contract,
)

ROOT = Path(__file__).resolve().parents[1]
PARENT_INVENTORY = (
    ROOT
    / "research/construct_validity/PIVOT_EXP_A3/validation/trial_inventory.jsonl.gz"
)


def test_a3r_continuation_contract_valid_cases() -> None:
    expected = {
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4",
        " 2 ": "2",
        "2\n": "2",
    }
    for raw, choice_id in expected.items():
        parsed = parse_continuation(raw)
        assert parsed.valid
        assert parsed.raw_valid
        assert parsed.reconstructed_valid
        assert parsed.choice_ids_match
        assert parsed.choice_id == choice_id
        assert parsed.raw_continuation == raw
        assert parsed.reconstructed_contract_output == PROMPT_COMPLETION_PREFIX + raw


def test_a3r_continuation_contract_invalid_cases() -> None:
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
    for raw in invalid:
        parsed = parse_continuation(raw)
        assert not parsed.valid
        assert parsed.choice_id is None


def test_a3r_reconstructed_forms_and_adversarial_inventory_pass() -> None:
    report = validate_parser_contract()
    assert report["passed"]
    assert [row["output"] for row in report["reconstructed_cases"]] == [
        "FINAL_CHOICE=1",
        "FINAL_CHOICE=2",
        "FINAL_CHOICE=3",
        "FINAL_CHOICE=4",
    ]


def test_a3_v1_parser_remains_strict_and_unchanged() -> None:
    assert not parse_final_choice("2").valid
    assert parse_final_choice("FINAL_CHOICE=2").valid


def test_a3r_only_amends_secondary_surface_boundary() -> None:
    primary = None
    secondary = None
    with gzip.open(PARENT_INVENTORY, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["response_method"] == "forced_choice" and primary is None:
                primary = row
            if row["response_method"] == "free_generation" and secondary is None:
                secondary = row
            if primary is not None and secondary is not None:
                break
    assert primary is not None and secondary is not None
    amended_primary = _amend_trial(primary)
    assert {**amended_primary, "study_id": primary["study_id"]} == primary

    amended_secondary = _amend_trial(secondary)
    assert amended_secondary["prompt"].endswith(PROMPT_COMPLETION_PREFIX)
    assert not amended_secondary["prompt"].endswith(PROMPT_COMPLETION_PREFIX + " ")
    assert f"ANSWER_CONTRACT={CONTRACT_VERSION}" in amended_secondary["prompt"]
    assert amended_secondary["prompt_sha256"] != secondary["prompt_sha256"]
    assert amended_secondary["trial_key"].endswith(
        f"response_contract={CONTRACT_VERSION}"
    )
    for field in (
        "image",
        "choices",
        "correct_choice_id",
        "scene_truth_choice_id",
        "declared_choice_id",
    ):
        assert amended_secondary[field] == secondary[field]


def test_a3r_config_preserves_registered_scientific_design() -> None:
    config = load_config()
    assert config["study"]["id"] == "PIVOT_EXP_A3R"
    assert config["study"]["validation_outcomes_observed"] is False
    assert config["study"]["scientific_metrics_observed"] is False
    assert config["answer_contract"]["primary_method"] == (
        "conditional_log_likelihood_single_token_argmax_v1"
    )
    assert config["gates"]["B_secondary_parser_integrity"] == {
        "parse_rate_minimum": 0.99,
        "confidence_interval_lower_minimum": 0.98,
        "raw_continuation_parse_rate_minimum": 0.99,
        "raw_continuation_ci_lower_minimum": 0.98,
        "reconstructed_contract_parse_rate_minimum": 0.99,
        "reconstructed_contract_ci_lower_minimum": 0.98,
    }


def test_a3r_development_smoke_covers_all_160_required_cells() -> None:
    rows = _required_development_smoke_rows(_read_development_records())
    assert len(rows) == 160
    assert len({row["trial_key"] for row in rows}) == 160
    cv1 = [row for row in rows if row["task"] == "answer_contract_comprehension"]
    assert len(cv1) == 32
    assert sum(row["image_context"] == "neutral_image" for row in cv1) == 16
    assert sum(row["image_context"] == "original_scene_image" for row in cv1) == 16
