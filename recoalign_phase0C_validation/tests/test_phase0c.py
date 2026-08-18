from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase0c.analysis import make_decision
from phase0c.config import ExperimentConfig
from phase0c.model import _choice_token_ids, relation_prompt


@pytest.fixture
def config() -> ExperimentConfig:
    return ExperimentConfig.from_json(ROOT / "configs" / "protocol.json", root=ROOT)


def test_registered_protocol_is_full_and_fixed(config: ExperimentConfig) -> None:
    config.validate()
    assert config.scientific_protocol_valid()
    assert config.primary_availability_layer == 32
    assert (config.primary_patch_layer, config.primary_patch_alpha) == (16, 1.0)
    assert config.choice_words == ("left", "right", "above", "below")
    assert config.answer_prefix == " "
    assert config.candidate_scoring == "teacher_forced_shared_answer_prefix"
    assert config.protocol_version == 3
    assert config.attention_collection == "separate_eager_diagnostic_forward"


def test_behavior_prompt_does_not_leak_relation() -> None:
    base = {
        "attribute1": "red",
        "object1": "circle",
        "attribute2": "blue",
        "object2": "square",
        "relation": "left",
    }
    counterfactual = dict(base, relation="right")
    assert relation_prompt(base) == relation_prompt(counterfactual)
    assert "left, right, above, or below" in relation_prompt(base)


def test_choices_are_scored_after_shared_tokenizer_prefix(config: ExperimentConfig) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_dir, use_fast=False)
    prefix_id, choice_ids = _choice_token_ids(
        tokenizer, config.choice_words, config.answer_prefix
    )
    assert prefix_id == 29871
    assert len(set(choice_ids.values())) == 4
    assert all(
        tokenizer.encode(config.answer_prefix + word, add_special_tokens=False)
        == [prefix_id, choice_ids[word]]
        for word in config.choice_words
    )


def test_decision_requires_every_gate(config: ExperimentConfig) -> None:
    rows = []
    for layer in config.layer_indices:
        rows.append(
            {
                "layer": layer,
                "accuracy": 0.90,
                "accuracy_ci_low": 0.85,
                "behavior_accuracy": 0.60,
                "availability_behavior_gap": 0.30,
                "gap_ci_low": 0.20,
                "behavior_error_n": 100,
                "error_subset_accuracy": 0.90,
                "error_subset_ci_low": 0.80,
                "probe_converged": True,
            }
        )
    patch = {
        "selected_trials": 24,
        "primary": {
            "positive_recovery": {"mean": 0.50, "ci_low": 0.25},
            "positive_minus_control": {"mean": 0.30, "ci_low": 0.10},
        },
    }
    integrity = {
        "behavior_complete": True,
        "attention_complete": True,
        "patch_complete": True,
        "patch_baseline_reproduced": True,
    }
    assert make_decision(config, rows, patch, integrity)["decision"] == "GO"
    rows[-1]["availability_behavior_gap"] = 0.149
    assert make_decision(config, rows, patch, integrity)["decision"] == "NO-GO"


def test_protocol_file_contains_no_result_dependent_choice(config: ExperimentConfig) -> None:
    payload = json.loads((ROOT / "configs" / "protocol.json").read_text(encoding="utf-8"))
    assert payload["primary_patch_layer"] == 16
    assert payload["max_patch_failures"] == 24
    assert "decision" not in payload
