from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from recoalign.construct_validity_primary.decision import (
    ALLOWED_OUTCOMES,
    adjudicate_primary_construct_validity,
)
from recoalign.construct_validity_primary.integrity import (
    A3R_FINAL_MANIFEST,
    A3R_RETIREMENT,
    scorer_boundary_report,
    verify_freeze,
    verify_parent_freezes,
)
from recoalign.construct_validity_primary.inventory import (
    DEVELOPMENT_INVENTORY,
    EXPECTED_PRIMARY_ROWS,
    INHERITED_INVENTORY,
    PRIMARY_METHOD,
    PROTECTED_FIELDS,
    REGISTERED_CHOICE_IDS,
    load_development_inventory,
    materialize_inherited_primary_inventory,
    read_jsonl_gz,
    validate_inventory_shape,
)
from recoalign.construct_validity_primary.power import calculate_primary_power
from recoalign.construct_validity_primary.primary_scoring import score_registered_options
from recoalign.construct_validity_primary.runner import load_config
from recoalign.construct_validity_primary.statistics import holm_correction, paired_tost
from recoalign.models.vlm.base import PreparedInput

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src/recoalign/construct_validity_primary"
LOCAL_PARENT_DATASET = (
    ROOT
    / "outputs/construct_validity/PIVOT_EXP_A3/inventory/validation/20260901/dataset.jsonl"
)


def _require_unpublished_parent_assets() -> None:
    if not LOCAL_PARENT_DATASET.is_file():
        pytest.skip("requires unpublished local A3 parent scene and rendered-image assets")


class _Backend:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def score_choice_log_likelihoods(
        self, prepared_input: PreparedInput, choice_ids: tuple[str, ...]
    ) -> dict[str, float]:
        assert prepared_input.prompt.endswith("FINAL_CHOICE=")
        assert tuple(choice_ids) == REGISTERED_CHOICE_IDS
        return self.scores


def _prepared() -> PreparedInput:
    return PreparedInput(
        prompt="question\nFINAL_CHOICE=",
        image="neutral.png",
        condition="test",
        setting="construct_validity_primary",
        input_tokens=2,
        evidence_tokens=2,
        semantic_units=4,
        relation_count=0,
        semantic_facts_sha256=None,
        serialization="M1",
        padding_units=0,
        token_match_delta=0,
    )


def _analysis(
    *,
    m1: dict[str, bool] | None = None,
    m2: dict[str, bool] | None = None,
    measurement_passed: bool = True,
) -> dict[str, Any]:
    default = {gate: False for gate in ("A", "C", "D", "E", "F")}

    def candidate(values: dict[str, bool] | None) -> dict[str, Any]:
        gates = {**default, **(values or {})}
        return {"gates": gates, "passed_all_gates": all(gates.values())}

    return {
        "completed_seeds": [20260901, 20260902, 20260903, 20260904, 20260905],
        "gate_A": {"passed": measurement_passed},
        "candidates": {"M1": candidate(m1), "M2": candidate(m2)},
    }


def _integrity(*, passed: bool = True, measurement_failure: bool = False) -> dict[str, Any]:
    return {"passed": passed, "primary_measurement_failure": measurement_failure}


def test_a3_v1_and_a3r_frozen_hashes_match() -> None:
    _require_unpublished_parent_assets()
    report = verify_parent_freezes()
    assert report["passed"] is True
    assert report["a3_v1_hash_mismatches"] == []
    assert report["a3r_hash_mismatches"] == []


def test_a3r_retirement_preserves_all_invalid_outputs() -> None:
    assert A3R_RETIREMENT.is_file()
    record = yaml.safe_load(A3R_RETIREMENT.read_text(encoding="utf-8"))
    assert record["secondary_instrument"]["invalid_outputs"] == {
        "E2": 25,
        "E4": 6,
        "E1": 2,
    }
    assert record["secondary_instrument"]["parsed_trials"] == 127
    assert record["secondary_instrument"]["unparsed_trials"] == 33


def test_a3r_final_manifest_has_required_roles_and_remote_commit() -> None:
    manifest = yaml.safe_load(A3R_FINAL_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["artifact_count"] >= 27
    assert set(manifest["required_roles"]) == {
        "preregistration",
        "amendment_rationale",
        "answer_contract",
        "development_smoke",
        "primary_immutability",
        "execution_stop",
        "next_stage_authorization",
        "trial_inventory",
        "secondary_prompt_hashes",
        "instrument_retirement",
    }
    assert len(manifest["source_remote_commit"]) == 40


def test_no_further_secondary_contract_is_registered() -> None:
    report = verify_parent_freezes()
    assert report["no_further_secondary_contract_registered"] is True


def test_inherited_inventory_has_exactly_27000_primary_rows() -> None:
    rows = read_jsonl_gz(INHERITED_INVENTORY)
    report = validate_inventory_shape(rows)
    assert len(rows) == EXPECTED_PRIMARY_ROWS
    assert report["passed"] is True
    assert report["by_seed"] == {seed: 5400 for seed in range(20260901, 20260906)}


def test_all_protected_fields_match_parent_rows_exactly() -> None:
    _require_unpublished_parent_assets()
    report = materialize_inherited_primary_inventory(allow_create=False)
    assert report["exact_protected_field_matches"] == EXPECTED_PRIMARY_ROWS
    assert report["mismatch_count"] == 0
    assert tuple(report["protected_fields"]) == PROTECTED_FIELDS
    assert report["passed"] is True


def test_trial_keys_prompts_images_choices_and_mappings_are_unchanged() -> None:
    _require_unpublished_parent_assets()
    report = materialize_inherited_primary_inventory(allow_create=False)
    assert report["trial_keys_changed"] is False
    assert report["image_asset_verification"]["verified_count"] == 2501
    assert report["image_asset_verification"]["hash_mismatches"] == []
    assert report["mismatches"] == []


def test_parent_image_sha_absence_is_protected_not_filled() -> None:
    rows = read_jsonl_gz(INHERITED_INVENTORY)
    assert all("image_sha256" not in row for row in rows)


def test_primary_scorer_signature_cannot_receive_ground_truth() -> None:
    signature = inspect.signature(score_registered_options)
    forbidden = {"correct_choice_id", "declared_choice_id", "scene_truth_choice_id", "answer"}
    assert not (set(signature.parameters) & forbidden)
    assert scorer_boundary_report()["passed"] is True


def test_primary_scorer_returns_four_finite_scores_and_registered_argmax() -> None:
    result = score_registered_options(
        _prepared(),
        REGISTERED_CHOICE_IDS,
        backend=_Backend({"1": -4.0, "2": -3.0, "3": -1.0, "4": -2.0}),
    )
    assert result.selected_option_id == "3"
    assert len(result.choice_log_likelihoods) == 4
    assert all(math.isfinite(value) for value in result.choice_log_likelihoods.values())
    assert result.method == PRIMARY_METHOD


def test_primary_scorer_rejects_nonfinite_or_incomplete_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        score_registered_options(
            _prepared(),
            REGISTERED_CHOICE_IDS,
            backend=_Backend({"1": 0.0, "2": -1.0, "3": float("nan"), "4": -2.0}),
        )
    with pytest.raises(ValueError, match="exactly four"):
        score_registered_options(
            _prepared(),
            REGISTERED_CHOICE_IDS,
            backend=_Backend({"1": 0.0, "2": -1.0, "3": -2.0}),
        )


def test_numeric_tie_break_is_retained() -> None:
    result = score_registered_options(
        _prepared(),
        REGISTERED_CHOICE_IDS,
        backend=_Backend({"1": -2.0, "2": -1.0, "3": -1.0, "4": -3.0}),
    )
    assert result.selected_option_id == "2"
    assert result.tied_maximum is True
    assert result.tie_state == "numeric_order_tiebreak"


def test_development_inventory_is_432_primary_only_rows() -> None:
    assert DEVELOPMENT_INVENTORY.is_file()
    rows = load_development_inventory()
    assert len(rows) == 432
    assert {row["seed"] for row in rows} == {20260830}
    assert _count_values(row["scene_id"] for row in rows) == {54}


def test_development_validation_and_a2_seeds_are_disjoint() -> None:
    development = {20260830}
    validation = set(range(20260901, 20260906))
    a2 = set(range(20260818, 20260823))
    assert not (development & validation)
    assert not (development & a2)
    assert not (validation & a2)


def test_primary_package_has_no_response_generation_call_path() -> None:
    for path in SOURCE_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr != "generate"


def test_primary_package_has_no_answer_interpretation_dependency() -> None:
    imports = []
    for path in SOURCE_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports.extend(
            node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        )
    assert all("answer_contract" not in module for module in imports)


def test_primary_scorer_has_no_model_update_api() -> None:
    source = (SOURCE_ROOT / "primary_scoring.py").read_text(encoding="utf-8")
    assert ".backward(" not in source
    assert ".step(" not in source
    assert "optimizer" not in source
    assert "inference_mode" in source


def test_power_is_recalculated_and_gate_f_reaches_target() -> None:
    report = calculate_primary_power()
    gate_f = report["endpoints"]["F_image_interference_per_task"]
    assert report["status"] == "PASS"
    assert report["secondary_generation_trials_removed"] == 9000
    assert report["primary_trial_count"] == 27000
    assert gate_f["estimated_power"] >= 0.80
    assert gate_f["most_demanding_endpoint"] is True


def test_holm_correction_matches_step_down_contract() -> None:
    result = holm_correction(
        {"shape": 0.001, "color": 0.01, "identity": 0.03, "binding": 0.2},
        alpha=0.05,
    )
    assert result["shape"]["reject"] is True
    assert result["binding"]["reject"] is False
    ordered = sorted(result.values(), key=lambda row: row["rank"])
    assert [row["threshold"] for row in ordered] == [0.0125, 0.05 / 3, 0.025, 0.05]


def test_paired_tost_synthetic_equivalence_contract() -> None:
    assert paired_tost([0.0] * 500, margin=0.05, alpha=0.05)["equivalent_unadjusted"]
    assert not paired_tost([0.10] * 500, margin=0.05, alpha=0.05)["equivalent_unadjusted"]


def test_secondary_gate_is_retired_not_passed_or_waived() -> None:
    config = load_config()
    assert config["secondary_gate"] == {
        "status": "RETIRED_BEFORE_VALIDATION",
        "participates_in_scientific_decision": False,
        "source_study": "PIVOT_EXP_A3R",
        "reason": "retired invalid external-validity instrument",
    }
    assert set(config["gates"]) == {
        "A_primary_answer_validity",
        "C_scaffold_comprehension",
        "D_scene_semantic_sufficiency",
        "E_intervention_separation",
        "F_image_interference",
    }


def test_m0_cannot_be_selected_and_both_pass_selects_m1() -> None:
    all_pass = {gate: True for gate in ("A", "C", "D", "E", "F")}
    result = adjudicate_primary_construct_validity(
        _analysis(m1=all_pass, m2=all_pass), integrity=_integrity()
    )
    assert result["outcome"] == "CONSTRUCT_VALIDITY_GO_TEXT"
    assert result["selected_manipulation"] == "M1"
    assert result["selected_manipulation"] != "M0"


def test_m2_is_selected_only_after_m1_failure() -> None:
    all_pass = {gate: True for gate in ("A", "C", "D", "E", "F")}
    result = adjudicate_primary_construct_validity(
        _analysis(m1={"A": True}, m2=all_pass), integrity=_integrity()
    )
    assert result["outcome"] == "CONSTRUCT_VALIDITY_GO_VISUAL_LEGEND"
    assert result["selected_manipulation"] == "M2"


def test_task_failure_cannot_be_hidden_by_aggregate() -> None:
    almost = {gate: True for gate in ("A", "C", "D", "E", "F")}
    almost["D"] = False
    result = adjudicate_primary_construct_validity(
        _analysis(m1=almost, m2={"A": True}), integrity=_integrity()
    )
    assert not result["outcome"].startswith("CONSTRUCT_VALIDITY_GO")


def test_primary_measurement_failure_has_specific_outcome() -> None:
    result = adjudicate_primary_construct_validity(
        _analysis(), integrity=_integrity(passed=False, measurement_failure=True)
    )
    assert result["outcome"] == "PRIMARY_MEASUREMENT_FAILURE"


def test_all_outcomes_prohibit_model_development_and_paper_writing() -> None:
    cases = [
        adjudicate_primary_construct_validity(_analysis(), integrity=_integrity(passed=False)),
        adjudicate_primary_construct_validity(
            _analysis(), integrity=_integrity(passed=False, measurement_failure=True)
        ),
    ]
    assert ALLOWED_OUTCOMES == {
        "CONSTRUCT_VALIDITY_GO_TEXT",
        "CONSTRUCT_VALIDITY_GO_VISUAL_LEGEND",
        "PRIMARY_MEASUREMENT_FAILURE",
        "SEMANTIC_MANIPULATION_FAILURE",
        "IMAGE_INTERFERENCE_DIAGNOSIS",
        "NO-GO_CONSTRUCT",
        "INCONCLUSIVE",
    }
    assert all(not row["authorization"]["model_development_allowed"] for row in cases)
    assert all(not row["authorization"]["paper_writing_allowed"] for row in cases)


def test_preinference_freeze_is_self_consistent_once_frozen() -> None:
    freeze = yaml.safe_load(
        (ROOT / "research/construct_validity/PIVOT_EXP_A3P/freeze_manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    if freeze["status"] == "FROZEN_PREINFERENCE":
        _require_unpublished_parent_assets()
        assert verify_freeze()["passed"] is True
        assert freeze["validation_inference_started"] is False


def _count_values(values: Any) -> set[int]:
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return set(counts.values())
