"""Immutable-parent, execution-boundary, and prediction integrity audits."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from . import primary_scoring
from .inventory import (
    EXPECTED_PRIMARY_ROWS,
    EXPECTED_SEEDS,
    PRIMARY_METHOD,
    REGISTERED_CHOICE_IDS,
    ROOT,
    STUDY_ROOT,
    sha256,
)

A3_ROOT = ROOT / "research/construct_validity/PIVOT_EXP_A3"
A3R_ROOT = ROOT / "research/construct_validity/PIVOT_EXP_A3R"
A3R_RETIREMENT = A3R_ROOT / "final_instrument_retirement_record.yaml"
A3R_FINAL_MANIFEST = A3R_ROOT / "final_artifact_manifest.yaml"
FREEZE_PATH = STUDY_ROOT / "freeze_manifest.yaml"


def write_yaml_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def create_secondary_retirement_record() -> dict[str, Any]:
    smoke_path = A3R_ROOT / "validation/development_runtime_smoke_report.yaml"
    smoke = yaml.safe_load(smoke_path.read_text(encoding="utf-8"))
    invalid = Counter(
        str(row["raw_continuation"])
        for row in smoke["trials"]
        if not bool(row["raw_continuation_parsed"])
    )
    record = {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A3R",
        "outcome": "SECONDARY_CONTRACT_RUNTIME_FAILURE",
        "validation_inference_started": False,
        "scientific_metrics_computed": False,
        "formal_adjudication_run": False,
        "secondary_instrument": {
            "name": "frozen_free_generation_option_contract",
            "status": "RETIRED_BEFORE_VALIDATION",
            "development_trials": 160,
            "parsed_trials": 127,
            "unparsed_trials": 33,
            "parse_rate": 0.79375,
            "deterministic_repeat_rate": 1.0,
            "invalid_outputs": {"E2": 25, "E4": 6, "E1": 2},
        },
        "observed_invalid_output_counts_verified": dict(invalid),
        "retirement_basis": [
            "PIVOT_EXP_A3 full-field development contract failure",
            "PIVOT_EXP_A3R continuation-contract development failure",
            "entity-ID and option-ID namespace confusion",
            "avoidance of repeated tuning and researcher degrees of freedom",
        ],
        "interpretation": {
            "model_failure_claim_allowed": False,
            "semantic_failure_claim_allowed": False,
            "measurement_instrument_valid": False,
            "further_prompt_repairs_allowed": False,
            "further_parser_repairs_allowed": False,
        },
        "next_study": "PIVOT_EXP_A3P",
        "next_study_scope": "independent preregistered primary-only construct validity",
        "no_validation_outcomes_observed": True,
        "no_primary_outcome_selected_before_retirement": True,
    }
    if invalid != Counter({"E2": 25, "E4": 6, "E1": 2}):
        raise ValueError(f"A3R invalid output evidence drifted: {invalid}")
    if A3R_RETIREMENT.exists():
        existing = yaml.safe_load(A3R_RETIREMENT.read_text(encoding="utf-8"))
        if existing != record:
            raise ValueError("existing A3R retirement record differs from frozen evidence")
    else:
        write_yaml_atomic(A3R_RETIREMENT, record)
    return record


def create_a3r_final_manifest() -> dict[str, Any]:
    create_secondary_retirement_record()
    artifacts = {}
    for path in sorted(A3R_ROOT.rglob("*")):
        if path.is_file() and path != A3R_FINAL_MANIFEST:
            artifacts[path.relative_to(A3R_ROOT).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    roles = {
        "preregistration": "preregistration.md",
        "amendment_rationale": "amendment_rationale.md",
        "answer_contract": "answer_contract/answer-contract-v3-continuation.yaml",
        "development_smoke": "validation/development_runtime_smoke_report.yaml",
        "primary_immutability": "validation/primary_immutability_report.yaml",
        "execution_stop": "validation/execution_stop_record.yaml",
        "next_stage_authorization": "validation/next_stage_authorization.yaml",
        "trial_inventory": "validation/trial_inventory.jsonl.gz",
        "secondary_prompt_hashes": "validation/secondary_prompt_rehash_report.yaml",
        "instrument_retirement": "final_instrument_retirement_record.yaml",
    }
    remote_commit = _git("rev-parse", "origin/codex/claim-evidence-execution")
    manifest = {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A3R",
        "outcome": "SECONDARY_CONTRACT_RUNTIME_FAILURE",
        "instrument_status": "RETIRED_BEFORE_VALIDATION",
        "hash_algorithm": "sha256",
        "source_remote_ref": "origin/codex/claim-evidence-execution",
        "source_remote_commit": remote_commit,
        "validation_inference_started": False,
        "scientific_metrics_computed": False,
        "formal_adjudication_run": False,
        "artifact_count": len(artifacts),
        "required_roles": {
            role: {"path": path, "sha256": artifacts[path]["sha256"]}
            for role, path in roles.items()
        },
        "artifacts": artifacts,
        "manifest_self_hash_policy": "excluded_to_avoid_circular_digest",
        "immutable": True,
    }
    if A3R_FINAL_MANIFEST.exists():
        existing = yaml.safe_load(A3R_FINAL_MANIFEST.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("existing A3R final manifest differs from current frozen artifacts")
    else:
        write_yaml_atomic(A3R_FINAL_MANIFEST, manifest)
    return manifest


def verify_parent_freezes() -> dict[str, Any]:
    v1_path = A3_ROOT / "v1_artifact_manifest.yaml"
    v1 = yaml.safe_load(v1_path.read_text(encoding="utf-8"))
    a3_mismatches = []
    for relative, expected in v1["artifacts"].items():
        path = A3_ROOT / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            a3_mismatches.append({"path": relative, "expected": expected, "actual": actual})
    inventory_manifest_path = A3_ROOT / "validation/inventory_manifest.yaml"
    inventory_manifest = yaml.safe_load(inventory_manifest_path.read_text(encoding="utf-8"))
    for relative, metadata in inventory_manifest["files"].items():
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        if actual != metadata["sha256"]:
            a3_mismatches.append(
                {
                    "path": relative,
                    "expected": metadata["sha256"],
                    "actual": actual,
                }
            )
    if not A3R_FINAL_MANIFEST.is_file():
        raise FileNotFoundError(A3R_FINAL_MANIFEST)
    a3r = yaml.safe_load(A3R_FINAL_MANIFEST.read_text(encoding="utf-8"))
    a3r_mismatches = []
    for relative, metadata in a3r["artifacts"].items():
        path = A3R_ROOT / relative
        actual = sha256(path) if path.is_file() else None
        if actual != metadata["sha256"]:
            a3r_mismatches.append(
                {"path": relative, "expected": metadata["sha256"], "actual": actual}
            )
    retirement = yaml.safe_load(A3R_RETIREMENT.read_text(encoding="utf-8"))
    registered_study_directories = {
        path.name
        for path in (ROOT / "research/construct_validity").iterdir()
        if path.is_dir() and path.name.startswith("PIVOT_EXP_A3")
    }
    contract_files = {
        path.name for path in (ROOT / "research/construct_validity").rglob("answer-contract-*.yaml")
    }
    no_new_contract = registered_study_directories <= {
        "PIVOT_EXP_A3",
        "PIVOT_EXP_A3R",
        "PIVOT_EXP_A3P",
    } and contract_files <= {
        "answer-contract-v2.yaml",
        "answer-contract-v3-continuation.yaml",
    }
    invalid_ok = retirement["secondary_instrument"]["invalid_outputs"] == {
        "E2": 25,
        "E4": 6,
        "E1": 2,
    }
    return {
        "a3_v1_manifest_sha256": sha256(v1_path),
        "a3_inventory_manifest_sha256": sha256(inventory_manifest_path),
        "a3_v1_hash_mismatches": a3_mismatches,
        "a3r_final_manifest_sha256": sha256(A3R_FINAL_MANIFEST),
        "a3r_hash_mismatches": a3r_mismatches,
        "a3r_invalid_outputs_preserved": invalid_ok,
        "secondary_retirement_record_exists": A3R_RETIREMENT.is_file(),
        "no_further_secondary_contract_registered": no_new_contract,
        "passed": not a3_mismatches and not a3r_mismatches and invalid_ok and no_new_contract,
    }


def scorer_boundary_report() -> dict[str, Any]:
    signature = inspect.signature(primary_scoring.score_registered_options)
    parameters = set(signature.parameters)
    forbidden = {"correct_choice_id", "declared_choice_id", "scene_truth_choice_id", "answer"}
    source = Path(primary_scoring.__file__).read_text(encoding="utf-8")
    checks = {
        "ground_truth_absent_from_signature": not (parameters & forbidden),
        "exact_registered_ids": primary_scoring.REGISTERED_CHOICE_IDS == REGISTERED_CHOICE_IDS,
        "single_registered_method": primary_scoring.PRIMARY_METHOD == PRIMARY_METHOD,
        "no_model_update_api": all(
            token not in source for token in ("optimizer", ".backward(", ".step(")
        ),
        "inference_mode_present": "inference_mode" in source,
    }
    return {"checks": checks, "passed": all(checks.values())}


def validate_prediction_integrity(
    predictions: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    *,
    freeze_sha256: str,
    environment_sha256: str,
) -> dict[str, Any]:
    expected = {str(row["trial_key"]): row for row in inventory}
    observed = {str(row.get("trial_key")): row for row in predictions}
    duplicate_count = len(predictions) - len(observed)
    by_seed = Counter(int(row.get("seed", -1)) for row in predictions)
    four_scores = 0
    finite_scores = 0
    registered = 0
    exact_rows = 0
    for row in predictions:
        trial = expected.get(str(row.get("trial_key")))
        scores = row.get("choice_log_likelihoods")
        four = isinstance(scores, dict) and tuple(scores) == REGISTERED_CHOICE_IDS
        finite = four and all(math.isfinite(float(value)) for value in scores.values())
        prediction_registered = str(row.get("selected_option_id")) in REGISTERED_CHOICE_IDS
        four_scores += int(four)
        finite_scores += int(finite)
        registered += int(prediction_registered)
        if trial and all(row.get(field) == trial.get(field) for field in trial):
            exact_rows += 1
    measurement_failure = not (
        len(predictions) == EXPECTED_PRIMARY_ROWS
        and four_scores == finite_scores == registered == EXPECTED_PRIMARY_ROWS
    )
    checks = {
        "prediction_count": len(predictions) == EXPECTED_PRIMARY_ROWS,
        "no_duplicates": duplicate_count == 0,
        "exact_trial_key_set": set(observed) == set(expected),
        "inventory_fields_preserved": exact_rows == EXPECTED_PRIMARY_ROWS,
        "four_scores_every_row": four_scores == EXPECTED_PRIMARY_ROWS,
        "finite_scores_every_row": finite_scores == EXPECTED_PRIMARY_ROWS,
        "registered_prediction_every_row": registered == EXPECTED_PRIMARY_ROWS,
        "scoring_method_frozen": all(
            row.get("scoring_method") == PRIMARY_METHOD for row in predictions
        ),
        "freeze_sha_frozen": all(row.get("freeze_sha256") == freeze_sha256 for row in predictions),
        "environment_sha_frozen": all(
            row.get("environment_sha256") == environment_sha256 for row in predictions
        ),
        "five_complete_seeds": by_seed == Counter({seed: 5400 for seed in EXPECTED_SEEDS}),
        "no_parameter_update": all(
            row.get("model_parameter_update") is False for row in predictions
        ),
    }
    return {
        "checks": checks,
        "prediction_count": len(predictions),
        "by_seed": dict(sorted(by_seed.items())),
        "duplicate_count": duplicate_count,
        "primary_measurement_failure": measurement_failure,
        "passed": all(checks.values()),
    }


def verify_freeze() -> dict[str, Any]:
    freeze = yaml.safe_load(FREEZE_PATH.read_text(encoding="utf-8"))
    mismatches = []
    for relative, expected in freeze["locked_artifacts"].items():
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            mismatches.append({"path": relative, "expected": expected, "actual": actual})
    required = {
        "status": "FROZEN_PREINFERENCE",
        "parent_primary_immutability": True,
        "primary_power_passed": True,
        "development_primary_smoke_passed": True,
        "validation_inventory_frozen": True,
        "execution_runtime_ready": True,
        "validation_inference_started": False,
    }
    fields_match = all(freeze.get(key) == value for key, value in required.items())
    return {
        "freeze_sha256": sha256(FREEZE_PATH),
        "locked_artifact_mismatches": mismatches,
        "required_fields_match": fields_match,
        "passed": not mismatches and fields_match,
    }


def git_execution_boundary() -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    remote = _git("rev-parse", "origin/codex/claim-evidence-execution")
    status = _git("status", "--porcelain")
    subject = _git("log", "-1", "--pretty=%s")
    return {
        "head": head,
        "remote": remote,
        "clean_worktree": status == "",
        "head_pushed": head == remote,
        "commit_subject": subject,
        "commit_1_present": subject
        == "Retire failed secondary instrument and preregister PIVOT_EXP_A3P",
        "passed": status == ""
        and head == remote
        and subject == "Retire failed secondary instrument and preregister PIVOT_EXP_A3P",
    }


def stable_environment_payload(model_provenance: dict[str, Any]) -> dict[str, Any]:
    import platform

    import bitsandbytes
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "transformers": str(transformers.__version__),
        "bitsandbytes": str(bitsandbytes.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model": model_provenance,
        "primary_method": PRIMARY_METHOD,
    }


def payload_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


__all__ = [
    "A3R_FINAL_MANIFEST",
    "A3R_RETIREMENT",
    "create_a3r_final_manifest",
    "create_secondary_retirement_record",
    "git_execution_boundary",
    "payload_sha256",
    "scorer_boundary_report",
    "stable_environment_payload",
    "validate_prediction_integrity",
    "verify_freeze",
    "verify_parent_freezes",
    "write_json_atomic",
    "write_yaml_atomic",
]
