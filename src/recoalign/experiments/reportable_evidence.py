"""Revalidate retained evidence before consuming reportable Winoground results."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from recoalign.data.manifest import sha256_file
from recoalign.experiments.records import (
    WINOGROUND_ARTIFACTS,
    load_run,
    validate_completed_winoground_run_integrity,
    validate_prediction_review,
)
from recoalign.experiments.run_comparison import (
    DECISION_FIELDS,
    METADATA_IDENTITY_FIELDS,
    RUN_IDENTITY_FIELDS,
)
from recoalign.experiments.winoground_audit import load_prediction_sample_ids


def validate_reportable_run_evidence(
    run_dir: str | Path,
    *,
    results_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate promotion evidence and the retained cache-disabled run."""
    directory = Path(run_dir)
    run = load_run(directory)
    if run["dataset"] != "winoground":
        return {"directory": directory, "run": run}
    if run["status"] != "reportable":
        raise ValueError("Winoground evidence validation requires status reportable")
    review = run.get("review")
    if not isinstance(review, dict) or review.get("decision") != "reportable":
        raise ValueError("reportable Winoground run requires reportable review metadata")

    evidence_path = _review_file(directory, review, "promotion_evidence")
    comparison_path = _review_file(directory, review, "comparison")
    prediction_review_path = _review_file(directory, review, "prediction_review")
    evidence = _load_json(evidence_path, "promotion evidence")
    comparison = _load_json(comparison_path, "run comparison")

    _require_equal(evidence, "canonical_run_id", run["run_id"])
    verification_run_id = review["verification_run_id"]
    _require_equal(evidence, "verification_run_id", verification_run_id)
    for field in (
        "comparison_file",
        "comparison_sha256",
        "prediction_review_file",
        "prediction_review_sha256",
        "prediction_review_count",
    ):
        _require_equal(evidence, field, review[field])
    if review["prediction_review_count"] != 400:
        raise ValueError("reportable Winoground prediction review count must be 400")
    for field, expected in (
        ("comparison_passed", True),
        ("mapping_checked_count", 400),
        ("prediction_review_count", 400),
    ):
        _require_equal(evidence, field, expected)

    audit = evidence.get("prediction_audit")
    if not isinstance(audit, dict):
        raise ValueError("promotion evidence prediction_audit must be an object")
    for field, expected in (
        ("prediction_count", 400),
        ("unique_sample_ids", 400),
        ("metrics_recomputed", True),
        ("decisions_recomputed", True),
        ("annotation_alignment_verified", True),
        ("all_recomputed_metrics_verified", True),
    ):
        _require_equal(audit, field, expected, prefix="prediction_audit.")
    count = audit.get("recomputed_metric_count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("prediction_audit.recomputed_metric_count must be positive")

    _validate_comparison(
        comparison,
        canonical_run_id=run["run_id"],
        verification_run_id=verification_run_id,
    )
    prediction_ids = load_prediction_sample_ids(directory)
    review_summary = validate_prediction_review(prediction_review_path, prediction_ids)
    if review_summary["prediction_review_count"] != 400:
        raise ValueError("prediction review must contain 400 rows")

    _require_equal(evidence, "canonical_config_sha256", run["config_sha256"])
    _require_equal(
        evidence,
        "canonical_dataset_manifest_sha256",
        run["dataset_manifest_sha256"],
    )
    _require_equal(
        evidence,
        "canonical_checkpoint_manifest_sha256",
        run["checkpoint_manifest_sha256"],
    )
    _validate_artifact_digests(
        directory,
        evidence.get("canonical_artifacts"),
        label="reportable canonical",
    )

    root = Path(results_root) if results_root is not None else directory.parent
    matches = _find_runs_by_id(root, verification_run_id)
    if not matches:
        raise ValueError("verification run is missing from results root")
    if len(matches) != 1:
        raise ValueError("multiple verification runs share the recorded run ID")
    verification_directory, verification_run = matches[0]
    if verification_run["status"] != "complete":
        raise ValueError("verification run is no longer complete")
    if verification_run["review"] is not None:
        raise ValueError("verification run review is no longer null")
    if verification_run["dataset"] != "winoground":
        raise ValueError("verification run dataset is no longer Winoground")
    expected_run_digest = evidence.get("verification_run_record_sha256")
    if not _is_sha256(expected_run_digest):
        raise ValueError("verification run record digest is missing or invalid")
    if sha256_file(verification_directory / "run.json") != expected_run_digest:
        raise ValueError("verification run record digest mismatch")
    _validate_artifact_digests(
        verification_directory,
        evidence.get("verification_artifacts"),
        label="verification",
    )
    verification_integrity = validate_completed_winoground_run_integrity(
        verification_directory,
        expected_cache_enabled=False,
    )
    return {
        "directory": directory,
        "run": run,
        "promotion_evidence": evidence,
        "comparison": comparison,
        "prediction_review": review_summary,
        "verification": verification_integrity,
    }


def _review_file(directory: Path, review: dict[str, Any], stem: str) -> Path:
    relative = review.get(f"{stem}_file")
    expected_digest = review.get(f"{stem}_sha256")
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(f"{stem} file path must be a non-empty relative path")
    posix = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in posix.parts
        or ".." in windows.parts
    ):
        raise ValueError(f"unsafe {stem} evidence path")
    path = directory / Path(relative)
    try:
        path.resolve().relative_to(directory.resolve())
    except (OSError, ValueError) as exc:
        raise ValueError(f"unsafe {stem} evidence path") from exc
    if not path.is_file():
        raise FileNotFoundError(f"{stem} evidence file is missing")
    if not _is_sha256(expected_digest):
        raise ValueError(f"{stem} SHA is missing or invalid")
    if sha256_file(path) != expected_digest:
        raise ValueError(f"{stem} SHA mismatch")
    return path


def _validate_comparison(
    comparison: dict[str, Any],
    *,
    canonical_run_id: str,
    verification_run_id: str,
) -> None:
    for field, expected in (
        ("passed", True),
        ("cached_run_id", canonical_run_id),
        ("no_cache_run_id", verification_run_id),
        ("cached_run_cache_enabled", True),
        ("no_cache_run_cache_disabled", True),
        ("cached_run_status_valid", True),
        ("no_cache_run_status_valid", True),
        ("metric_names_match", True),
        ("metrics_within_tolerance", True),
        ("prediction_count_match", True),
        ("sample_id_order_match", True),
        ("scores_within_tolerance", True),
    ):
        _require_equal(comparison, field, expected)
    expected_gate_fields = {
        "run_identity_matches": set(RUN_IDENTITY_FIELDS),
        "metadata_identity_matches": set(METADATA_IDENTITY_FIELDS),
    }
    for field, expected_fields in expected_gate_fields.items():
        gates = comparison.get(field)
        if (
            not isinstance(gates, dict)
            or set(gates) != expected_fields
            or any(value is not True for value in gates.values())
        ):
            raise ValueError(f"run comparison gate failed: {field}")
    differences = comparison.get("decision_differences")
    if (
        not isinstance(differences, dict)
        or set(differences) != set(DECISION_FIELDS)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value != 0
            for value in differences.values()
        )
    ):
        raise ValueError("run comparison gate failed: decision_differences")


def _validate_artifact_digests(
    directory: Path,
    artifact_digests: Any,
    *,
    label: str,
) -> None:
    if not isinstance(artifact_digests, dict):
        raise ValueError(f"{label} artifact digests must be an object")
    if set(artifact_digests) != set(WINOGROUND_ARTIFACTS):
        raise ValueError(f"{label} artifact digest inventory is incomplete")
    for relative in WINOGROUND_ARTIFACTS:
        digest = artifact_digests.get(relative)
        if not _is_sha256(digest):
            raise ValueError(f"{label} artifact digest is invalid: {relative}")
        path = directory / relative
        if not path.is_file():
            raise FileNotFoundError(f"{label} artifact is missing: {relative}")
        if sha256_file(path) != digest:
            raise ValueError(f"{label} artifact digest mismatch: {relative}")


def _find_runs_by_id(root: Path, run_id: str) -> list[tuple[Path, dict[str, Any]]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for run_file in sorted(root.glob("**/run.json")):
        payload = _load_json(run_file, "run")
        if payload.get("run_id") == run_id:
            matches.append((run_file.parent, payload))
    return matches


def _require_equal(
    payload: dict[str, Any],
    field: str,
    expected: Any,
    *,
    prefix: str = "",
) -> None:
    if payload.get(field) != expected:
        raise ValueError(f"promotion evidence field mismatch: {prefix}{field}")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
