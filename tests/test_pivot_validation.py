from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pytest

from recoalign.cli import build_parser
from recoalign.models.vlm.base import ReferenceVLM
from recoalign.models.vlm.registry import ModelRegistry
from recoalign.pivot_validation.analysis import classify_mechanism
from recoalign.pivot_validation.design import (
    A1_TASKS,
    A2_CONDITIONS,
    A3_CONDITIONS,
    QUESTION_TYPES,
    _canonical_relation_facts,
    _object_facts,
    build_seed_trials,
    load_source_records,
    select_balanced_records,
    validate_seed_trials,
)
from recoalign.pivot_validation.runner import load_pivot_config
from recoalign.synthetic_world.questions import canonical_facts

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260818
SOURCE = ROOT / f"outputs/paper_evidence/EXP001/seeds/{SEED}/dataset/dataset.jsonl"


@pytest.fixture(scope="module")
def source_records() -> list[Any]:
    return select_balanced_records(load_source_records(SOURCE), 40)


@pytest.fixture(scope="module")
def reference_trials(source_records: list[Any]) -> list[Any]:
    return build_seed_trials(ReferenceVLM(seed=SEED), source_records, seed=SEED)


def test_balanced_selection_has_ten_scenes_per_question_type(
    source_records: list[Any],
) -> None:
    counts = Counter(str(record.metadata["question_type"]) for record in source_records)
    assert counts == Counter({question_type: 10 for question_type in QUESTION_TYPES})
    assert len({record.scene_id for record in source_records}) == 40


def test_a1_is_four_choice_without_question_answer_leakage_and_balanced(
    reference_trials: list[Any],
) -> None:
    a1 = [trial for trial in reference_trials if trial.experiment == "A1"]
    assert len(a1) == 40 * len(A1_TASKS)
    positions: dict[str, Counter[int]] = defaultdict(Counter)
    for trial in a1:
        assert len(trial.record.choices) == 4
        assert trial.record.choices.count(trial.record.answer) == 1
        question_tokens = re.findall(r"[a-z0-9_-]+", trial.record.question.casefold())
        assert trial.record.answer.casefold() not in question_tokens
        positions[trial.task][trial.metadata["correct_choice_index"]] += 1
    assert set(positions) == set(A1_TASKS)
    assert all(counts == Counter({0: 10, 1: 10, 2: 10, 3: 10}) for counts in positions.values())


def test_a2_factorization_covers_complete_fact_set(
    source_records: list[Any], reference_trials: list[Any]
) -> None:
    by_scene = {record.scene_id: record for record in source_records}
    a2 = [trial for trial in reference_trials if trial.experiment == "A2"]
    assert Counter(trial.condition for trial in a2) == Counter(
        {condition: 40 for condition in A2_CONDITIONS}
    )
    for record in by_scene.values():
        object_facts = set(_object_facts(record))
        relation_facts = set(_canonical_relation_facts(record))
        complete_facts = set(canonical_facts(record.objects, record.relations))
        assert object_facts.isdisjoint(relation_facts)
        assert object_facts | relation_facts == complete_facts


def test_a2_relation_evidence_does_not_leak_shape_or_color_answer(
    reference_trials: list[Any],
) -> None:
    relevant = [
        trial
        for trial in reference_trials
        if trial.experiment == "A2"
        and trial.condition == "relation_evidence"
        and trial.record.metadata["query"]["answer_field"] in {"shape", "color"}
    ]
    assert relevant
    for trial in relevant:
        evidence = trial.prepared.prompt.split("\nQuestion:", maxsplit=1)[0].casefold()
        evidence_tokens = re.findall(r"[a-z0-9_-]+", evidence)
        assert trial.record.answer.casefold() not in evidence_tokens


def test_a3_formats_share_one_fact_hash_and_design_integrity_passes(
    reference_trials: list[Any],
) -> None:
    grouped: dict[str, set[str | None]] = defaultdict(set)
    for trial in reference_trials:
        if trial.experiment == "A3":
            grouped[trial.source_scene_id].add(trial.prepared.semantic_facts_sha256)
    assert len(grouped) == 40
    assert all(len(hashes) == 1 and None not in hashes for hashes in grouped.values())
    report = validate_seed_trials(reference_trials, samples_per_seed=40)
    assert report["passed"] is True
    assert all(report["assertions"].values())


def test_llava_tokenizer_matches_a2_and_a3_without_loading_weights(
    source_records: list[Any],
) -> None:
    registry = ModelRegistry(ROOT)
    definition = registry.definition("llava_1_5_7b")
    model = registry.get_or_create(definition.experiment_model_config(), seed=SEED)
    trials = build_seed_trials(model, source_records[:1], seed=SEED)
    assert model.loaded is False
    for experiment, expected_conditions in (("A2", A2_CONDITIONS), ("A3", A3_CONDITIONS)):
        selected = [trial for trial in trials if trial.experiment == experiment]
        assert {trial.condition for trial in selected} == set(expected_conditions)
        counts = [trial.prepared.input_tokens for trial in selected]
        assert max(counts) - min(counts) <= 1


def _classification_metrics(
    config: dict[str, Any],
    *,
    object_status: str,
    attribute_status: str,
    relation_status: str,
    format_status: str,
    complete_stable: bool,
    relation_stable: bool = False,
    image_accuracy: float = 0.50,
    complete_accuracy: float = 0.80,
) -> dict[str, Any]:
    return {
        "completed_seeds": list(config["study"]["seeds"]),
        "integrity_passed": True,
        "A1": {
            "tasks": {
                "object_shape": {"availability": object_status},
                "attribute_color": {"availability": attribute_status},
                "direct_relation": {"availability": relation_status},
            }
        },
        "A2": {
            "stable_effects": {
                "object_contribution": False,
                "relation_contribution": relation_stable,
                "complete_contribution": complete_stable,
                "integration_surplus": False,
                "additive_synergy": False,
            },
            "conditions": {
                "image_only": {"summary": {"mean": image_accuracy}},
                "complete_evidence": {"summary": {"mean": complete_accuracy}},
            },
        },
        "A3": {"format_classification": format_status},
    }


@pytest.mark.parametrize(
    (
        "statuses",
        "format_status",
        "complete_stable",
        "expected_classification",
        "expected_decision",
    ),
    [
        (
            ("LOW", "LOW", "LOW"),
            "INCONCLUSIVE",
            False,
            "PRIMITIVE_SEMANTIC_REPRESENTATION_LIMITATION",
            "GO",
        ),
        (
            ("HIGH", "HIGH", "LOW"),
            "INCONCLUSIVE",
            True,
            "RELATIONAL_GROUNDING_FAILURE",
            "GO",
        ),
        (
            ("HIGH", "HIGH", "HIGH"),
            "FORMAT_SENSITIVE",
            False,
            "FORMAT_CONDITIONED_RELATIONAL_INTEGRATION_FAILURE",
            "GO",
        ),
        (
            ("HIGH", "HIGH", "HIGH"),
            "FORMAT_INVARIANT",
            True,
            "COMPOSITIONAL_REASONING_BOTTLENECK",
            "GO",
        ),
        (
            ("INDETERMINATE", "HIGH", "LOW"),
            "INCONCLUSIVE",
            False,
            "MULTIPLE_EXPLANATIONS_REMAIN",
            "NO-GO",
        ),
    ],
)
def test_registered_mechanism_classifications(
    statuses: tuple[str, str, str],
    format_status: str,
    complete_stable: bool,
    expected_classification: str,
    expected_decision: str,
) -> None:
    config = load_pivot_config()
    metrics = _classification_metrics(
        config,
        object_status=statuses[0],
        attribute_status=statuses[1],
        relation_status=statuses[2],
        format_status=format_status,
        complete_stable=complete_stable,
    )
    result = classify_mechanism(metrics, config)
    assert result["classification"] == expected_classification
    assert result["decision"] == expected_decision
    assert result["paper_writing_allowed"] is False
    assert result["model_development_allowed"] is False


def test_pivot_cli_commands_are_registered() -> None:
    parser = build_parser()
    validate = parser.parse_args(["validate-pivot"])
    pilot = parser.parse_args(["run-pivot-validation", "--stage", "pilot"])
    final = parser.parse_args(["run-pivot-validation", "--stage", "final"])
    assert validate.command == "validate-pivot"
    assert pilot.command == final.command == "run-pivot-validation"
    assert pilot.stage == "pilot"
    assert final.stage == "final"
