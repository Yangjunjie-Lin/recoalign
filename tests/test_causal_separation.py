from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
import yaml

from recoalign.causal_separation.decision import ALLOWED_OUTCOMES, adjudicate_mechanism
from recoalign.causal_separation.factorial_builder import (
    PRIMARY_SERIALIZATIONS,
    RELATION_CONDITIONS,
    SEMANTIC_CONDITIONS,
    build_seed_trials,
    load_source_records,
    validate_seed_trials,
)
from recoalign.causal_separation.integrity import (
    artifact_metadata,
    validate_artifact_manifest,
    validate_parent_freeze,
    validate_prediction_rows,
)
from recoalign.causal_separation.power import (
    build_power_analysis,
    paired_normal_power,
    tost_equivalence_power,
)
from recoalign.causal_separation.runner import load_causal_config
from recoalign.causal_separation.statistics import (
    hierarchical_summary,
    holm_correction,
    paired_tost,
)
from recoalign.cli import build_parser
from recoalign.models.vlm.base import ReferenceVLM
from recoalign.models.vlm.registry import ModelRegistry

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260818
SOURCE = ROOT / f"outputs/paper_evidence/EXP001/seeds/{SEED}/dataset/dataset.jsonl"
ALIAS_SALT = "pivot-exp-a2-alias-v1"
CORRUPTION_SALT = "pivot-exp-a2-corruption-v1"


@pytest.fixture(scope="module")
def source_records() -> list[Any]:
    if not SOURCE.is_file():
        pytest.skip("requires the unpublished local EXP001 source-run dataset")
    return load_source_records(SOURCE)


@pytest.fixture(scope="module")
def reference_trials(source_records: list[Any]) -> list[Any]:
    return build_seed_trials(
        ReferenceVLM(seed=SEED),
        source_records,
        seed=SEED,
        alias_salt=ALIAS_SALT,
        corruption_salt=CORRUPTION_SALT,
    )


def test_oracle_scaffold_matches_true_scene_attributes(reference_trials: list[Any]) -> None:
    main = [trial for trial in reference_trials if trial.family == "main"]
    for trial in main:
        truth = {
            str(node["id"]): (str(node["shape"]), str(node["color"]))
            for node in trial.record.objects
        }
        aliases = trial.metadata["alias_by_object_id"]
        oracle = {
            parts[1]: (parts[2].split("=", 1)[1], parts[3].split("=", 1)[1])
            for fact in trial.metadata["oracle_semantic_facts"]
            for parts in (fact.split("|"),)
        }
        assert all(oracle[aliases[identifier]] == values for identifier, values in truth.items())


def test_corrupted_scaffold_is_false_and_token_matched(reference_trials: list[Any]) -> None:
    main = [trial for trial in reference_trials if trial.family == "main"]
    assert all(trial.metadata["semantic_corruption_false"] for trial in main)
    grouped: dict[str, list[int]] = defaultdict(list)
    for trial in main:
        grouped[trial.source_scene_id].append(trial.prepared.input_tokens)
    assert all(max(values) - min(values) <= 1 for values in grouped.values())


def test_semantic_scaffold_has_no_relation_query_role_or_explicit_answer(
    reference_trials: list[Any],
) -> None:
    main = [trial for trial in reference_trials if trial.family == "main"]
    for trial in main:
        assert all(
            not fact.startswith("relation|")
            for fact in trial.metadata["active_semantic_facts"]
        )
        evidence = trial.prepared.prompt.split("\nQuestion:", maxsplit=1)[0]
        assert "answer_object" not in evidence
        assert "question_subject" not in evidence
        assert "final_answer" not in evidence
        assert "Answer:" not in evidence


def test_relation_corruption_preserves_semantic_facts_and_is_false(
    reference_trials: list[Any],
) -> None:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(dict)
    for trial in reference_trials:
        if trial.family == "main":
            key = (
                trial.source_scene_id,
                str(trial.factors["semantic"]),
                str(trial.factors["serialization"]),
            )
            grouped[key][str(trial.factors["relation"])] = trial
    assert grouped
    for pair in grouped.values():
        correct = pair["correct_relation"]
        corrupted = pair["corrupted_relation"]
        assert correct.metadata["active_semantic_facts"] == corrupted.metadata[
            "active_semantic_facts"
        ]
        assert corrupted.metadata["relation_corruption_false"] is True
        assert set(correct.metadata["active_relation_facts"]).isdisjoint(
            corrupted.metadata["active_relation_facts"]
        )


def test_json_and_triples_have_identical_declared_fact_hashes(
    reference_trials: list[Any],
) -> None:
    grouped: dict[tuple[str, str, str], set[str | None]] = defaultdict(set)
    for trial in reference_trials:
        if trial.family == "main":
            grouped[
                (
                    trial.source_scene_id,
                    str(trial.factors["semantic"]),
                    str(trial.factors["relation"]),
                )
            ].add(trial.prepared.semantic_facts_sha256)
    assert len(grouped) == 90 * 4
    assert all(len(hashes) == 1 and None not in hashes for hashes in grouped.values())


def test_each_scene_has_complete_eight_cell_matrix_and_unique_keys(
    reference_trials: list[Any],
) -> None:
    report = validate_seed_trials(reference_trials)
    assert report["passed"] is True
    assert all(report["assertions"].values())
    main = [trial for trial in reference_trials if trial.family == "main"]
    expected = {
        (semantic, relation, serialization)
        for semantic in SEMANTIC_CONDITIONS
        for relation in RELATION_CONDITIONS
        for serialization in PRIMARY_SERIALIZATIONS
    }
    grouped: dict[str, set[tuple[str | None, str | None, str | None]]] = defaultdict(set)
    for trial in main:
        grouped[trial.source_scene_id].add(
            (
                trial.factors["semantic"],
                trial.factors["relation"],
                trial.factors["serialization"],
            )
        )
    assert all(cells == expected for cells in grouped.values())


def test_manipulation_checks_are_independent_and_relation_free(
    reference_trials: list[Any],
) -> None:
    manipulation = [
        trial for trial in reference_trials if trial.family == "manipulation_check"
    ]
    assert len(manipulation) == 160
    assert all(trial.prepared.relation_count == 0 for trial in manipulation)
    assert all(trial.metadata["primary_analysis_eligible"] is False for trial in manipulation)
    assert {trial.task for trial in manipulation} == {
        "object_identity",
        "shape",
        "color",
        "entity_attribute_binding",
    }
    answer_positions = {
        trial.record.choices.index(trial.record.answer) for trial in manipulation
    }
    assert answer_positions == {0, 1, 2, 3}


def test_real_llava_tokenizer_obeys_frozen_tolerance_without_loading_weights(
    source_records: list[Any],
) -> None:
    registry = ModelRegistry(ROOT)
    definition = registry.definition("llava_1_5_7b")
    model = registry.get_or_create(definition.experiment_model_config(), seed=SEED)
    trials = build_seed_trials(
        model,
        source_records,
        seed=SEED,
        alias_salt=ALIAS_SALT,
        corruption_salt=CORRUPTION_SALT,
    )
    assert model.loaded is False
    assert validate_seed_trials(trials)["assertions"]["main_token_matched"] is True
    assert validate_seed_trials(trials)["assertions"]["manipulation_token_matched"] is True


def test_generation_protocol_is_identical(reference_trials: list[Any]) -> None:
    hashes = {trial.prepared.prompt_protocol_sha256 for trial in reference_trials}
    assert len(hashes) == 1
    assert None not in hashes


def test_power_analysis_is_frozen_from_parent_and_passes() -> None:
    config = load_causal_config()
    result = build_power_analysis(
        ROOT / "research/pivot_validation/results/predictions.jsonl.gz",
        planned_scenes=450,
        superiority_sesoi=config["statistics"]["superiority_sesoi"],
        equivalence_margin=config["statistics"]["equivalence_margin"],
        alpha=config["statistics"]["alpha"],
        target_power=config["statistics"]["target_power"],
    )
    assert result["status"] == "PASS"
    assert result["computed_before_inference"] is True
    assert all(
        value["superiority_power_at_sesoi"] >= 0.80
        and value["tost_power_if_true_effect_zero"] >= 0.80
        for value in result["estimands"].values()
    )
    assert paired_normal_power(450, effect=0.10, discordance=0.295, alpha=0.025) > 0.80
    assert tost_equivalence_power(450, margin=0.10, discordance=0.485, alpha=0.05) > 0.80


def test_parent_pivot_exp_a_assets_are_immutable() -> None:
    result = validate_parent_freeze(
        ROOT, "research/pivot_validation/results/pivot_exp_a_final_freeze.yaml"
    )
    assert result["passed"] is True
    assert all(result["assertions"].values())
    assert all(result["blob_match"].values())


def test_prediction_integrity_requires_five_equal_seeds_and_expected_lines() -> None:
    config = load_causal_config()
    freeze = {"freeze_name": "synthetic-freeze"}
    rows: list[dict[str, Any]] = []
    for seed in config["study"]["seeds"]:
        for scene in range(90):
            for cell in range(8):
                rows.append(
                    {
                        "seed": seed,
                        "source_scene_id": f"scene-{scene}",
                        "family": "main",
                        "trial_key": f"{seed}:main:{scene}:{cell}",
                        "freeze_name": "synthetic-freeze",
                        "input": {"input_tokens": 100 + cell % 2},
                        "metadata": {
                            "semantic_corruption_false": True,
                            "relation_corruption_false": True,
                        },
                        "prediction": "left",
                        "evaluation": {"matched_choice": "left"},
                    }
                )
        for index in range(340):
            rows.append(
                {
                    "seed": seed,
                    "source_scene_id": f"aux-{index}",
                    "family": "reference",
                    "trial_key": f"{seed}:aux:{index}",
                    "freeze_name": "synthetic-freeze",
                    "input": {"input_tokens": 10},
                    "metadata": {},
                    "prediction": "left",
                    "evaluation": {"matched_choice": "left"},
                }
            )
    result = validate_prediction_rows(rows, config, freeze)
    assert result["passed"] is True
    assert len(rows) == 5300


def test_artifact_sha256_validation(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    payload = result_dir / "metrics.json"
    payload.write_text("{}\n", encoding="utf-8")
    import gzip

    with gzip.open(result_dir / "predictions.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write("{}\n")
    manifest = {
        "prediction_lines": 1,
        "artifacts": {
            "metrics.json": artifact_metadata(payload),
            "predictions.jsonl.gz": artifact_metadata(result_dir / "predictions.jsonl.gz"),
        },
    }
    (result_dir / "artifact_manifest.yaml").write_text(
        yaml.safe_dump(manifest), encoding="utf-8"
    )
    assert validate_artifact_manifest(result_dir)["passed"] is True
    payload.write_text("changed\n", encoding="utf-8")
    assert validate_artifact_manifest(result_dir)["passed"] is False


def _summary(
    mean: float, *, lower: float, upper: float, equivalent: bool = False
) -> dict[str, Any]:
    positive = 1.0 if mean > 0 else 0.0
    negative = 1.0 if mean < 0 else 0.0
    return {
        "mean": mean,
        "confidence_interval": {"lower": lower, "upper": upper},
        "positive_seed_fraction": positive,
        "negative_seed_fraction": negative,
        "holm": {"reject_zero": lower > 0 or upper < 0},
        "tost": {"equivalent": equivalent},
    }


def _synthetic_analysis(
    *,
    manipulation: bool = True,
    e1: dict[str, Any],
    e2: dict[str, Any],
    e3: dict[str, Any],
    e4: dict[str, Any],
    e5: dict[str, Any],
) -> dict[str, Any]:
    return {
        "completed_seeds": [1, 2, 3, 4, 5],
        "manipulation_check": {"passed": manipulation},
        "estimands": {
            "E1_semantic_rescue": e1,
            "E2_relation_under_oracle": e2,
            "E3_json_minus_triples": e3,
            "E4_semantic_x_relation": e4,
            "E5_semantic_x_format": e5,
        },
        "hop_depth": {
            hop: {
                "E2_relation_under_oracle": e2,
                "E3_json_minus_triples": e3,
            }
            for hop in ("1", "2", "3", "4")
        },
    }


@pytest.mark.parametrize(
    ("analysis", "expected"),
    [
        (
            _synthetic_analysis(
                e1=_summary(0.20, lower=0.15, upper=0.25),
                e2=_summary(0.15, lower=0.10, upper=0.20),
                e3=_summary(0.12, lower=0.08, upper=0.16),
                e4=_summary(0.06, lower=0.02, upper=0.10),
                e5=_summary(0.01, lower=-0.03, upper=0.05),
            ),
            "JOINT_CAUSAL_GO",
        ),
        (
            _synthetic_analysis(
                e1=_summary(0.20, lower=0.15, upper=0.25),
                e2=_summary(0.01, lower=-0.03, upper=0.04, equivalent=True),
                e3=_summary(0.02, lower=-0.02, upper=0.05, equivalent=True),
                e4=_summary(0.00, lower=-0.04, upper=0.04),
                e5=_summary(0.00, lower=-0.04, upper=0.04),
            ),
            "SEMANTIC_PRIMARY_GO",
        ),
        (
            _synthetic_analysis(
                e1=_summary(0.03, lower=-0.01, upper=0.07),
                e2=_summary(0.15, lower=0.10, upper=0.20),
                e3=_summary(0.02, lower=-0.02, upper=0.06),
                e4=_summary(0.01, lower=-0.03, upper=0.05),
                e5=_summary(0.00, lower=-0.04, upper=0.04),
            ),
            "INTEGRATION_INDEPENDENT_GO",
        ),
        (
            _synthetic_analysis(
                manipulation=False,
                e1=_summary(0.20, lower=0.15, upper=0.25),
                e2=_summary(0.15, lower=0.10, upper=0.20),
                e3=_summary(0.12, lower=0.08, upper=0.16),
                e4=_summary(0.06, lower=0.02, upper=0.10),
                e5=_summary(0.01, lower=-0.03, upper=0.05),
            ),
            "INCONCLUSIVE",
        ),
    ],
)
def test_decision_engine_expresses_registered_mechanisms(
    analysis: dict[str, Any], expected: str
) -> None:
    result = adjudicate_mechanism(
        analysis, load_causal_config(), integrity_passed=True, power_passed=True
    )
    assert result["outcome"] == expected
    assert result["outcome"] in ALLOWED_OUTCOMES
    assert result["authorization"]["model_development_allowed"] is False
    assert result["authorization"]["paper_writing_allowed"] is False


def test_tost_holm_and_hierarchical_bootstrap_on_synthetic_data() -> None:
    equivalent = paired_tost([0.0] * 100, margin=0.10, alpha=0.05)
    non_equivalent = paired_tost([0.20] * 100, margin=0.10, alpha=0.05)
    assert equivalent["equivalent"] is True
    assert non_equivalent["equivalent"] is False
    holm = holm_correction({"E2": 0.01, "E3": 0.04}, alpha=0.05)
    assert holm["E2"]["reject_zero"] is True
    assert holm["E3"]["reject_zero"] is True
    config = load_causal_config()
    config = json.loads(json.dumps(config))
    config["statistics"]["bootstrap_samples"] = 1000
    values = {(seed, f"scene-{index}"): 0.20 for seed in range(5) for index in range(20)}
    summary = hierarchical_summary(values, config, offset=999)
    assert summary["mean"] == pytest.approx(0.20)
    assert summary["confidence_interval"]["lower"] == pytest.approx(0.20)


def test_cli_registers_all_causal_separation_lifecycle_commands() -> None:
    parser = build_parser()
    preregister = parser.parse_args(
        ["preregister-causal-separation", "--study", "PIVOT_EXP_A2"]
    )
    validate = parser.parse_args(
        [
            "validate-causal-separation",
            "--study",
            "PIVOT_EXP_A2",
            "--preflight-only",
        ]
    )
    run = parser.parse_args(
        [
            "run-causal-separation",
            "--study",
            "PIVOT_EXP_A2",
            "--model",
            "llava_1_5_7b",
        ]
    )
    adjudicate = parser.parse_args(
        ["adjudicate-causal-separation", "--study", "PIVOT_EXP_A2"]
    )
    assert preregister.command == "preregister-causal-separation"
    assert validate.preflight_only is True
    assert run.model == "llava_1_5_7b"
    assert adjudicate.command == "adjudicate-causal-separation"


def test_power_analysis_file_hash_is_stable() -> None:
    path = ROOT / "research/causal_separation/PIVOT_EXP_A2/power_analysis.yaml"
    first = hashlib.sha256(path.read_bytes()).hexdigest()
    second = hashlib.sha256(path.read_bytes()).hexdigest()
    assert first == second
