"""Create, finalize, fail, and promote self-contained experiment records."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from recoalign.config import config_digest, file_digest, validate_config
from recoalign.data.manifest import (
    load_checkpoint_manifest,
    load_dataset_manifest,
    sha256_file,
    snapshot_manifest,
    verify_annotation_inventory_coverage,
    verify_hashed_image_inventory,
    verify_manifest_files,
)
from recoalign.experiments.run_comparison import compare_runs
from recoalign.experiments.winoground_audit import (
    audit_winoground_run,
    load_prediction_sample_ids,
)
from recoalign.reproducibility import atomic_write_json, collect_environment, utc_now
from recoalign.schema_validation import repository_root, validate_payload

RUN_STATUSES = {"pilot", "partial", "failed", "complete", "reportable"}
FINALIZABLE_STATUSES = RUN_STATUSES - {"reportable"}
WINOGROUND_CONTENT_CHECK_METHOD = "casefolded_alphanumeric_character_multiset_v1"
WINOGROUND_ARTIFACTS = (
    "config.resolved.yaml",
    "environment.json",
    "manifests/dataset.yaml",
    "manifests/checkpoint.yaml",
    "evaluation.json",
    "metrics.json",
    "predictions.jsonl",
)


def create_run(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    output_root: str | Path | None = None,
    run_id: str | None = None,
) -> Path:
    """Create a run directory with config, manifests, environment, and provenance."""
    validate_config(config)
    digest = config_digest(config)
    name = _safe_component(config["experiment"]["name"])
    timestamp = utc_now().replace(":", "").replace("+00:00", "Z")
    identifier = run_id or f"{name}-{timestamp}-{digest[:8]}"
    root = Path(output_root or config["experiment"]["output_dir"])
    run_dir = root / identifier
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")

    project_root = repository_root()
    dataset_manifest_path = _resolve_path(config["data"]["manifest"], project_root)
    checkpoint_manifest_path = _resolve_path(config["model"]["manifest"], project_root)
    dataset_manifest = load_dataset_manifest(dataset_manifest_path)
    checkpoint_manifest = load_checkpoint_manifest(checkpoint_manifest_path)
    _validate_dataset_identity(config, dataset_manifest)

    run_dir.mkdir(parents=True)

    dataset_root = _resolve_path(config["data"]["root"], project_root)
    checkpoint_root = _resolve_path(config["model"].get("checkpoint_root", "."), project_root)
    dataset_verification = _verification_record(dataset_root, dataset_manifest)
    checkpoint_verification = _verification_record(checkpoint_root, checkpoint_manifest)

    with (run_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    snapshot_manifest(dataset_manifest_path, run_dir / "manifests" / "dataset.yaml")
    snapshot_manifest(checkpoint_manifest_path, run_dir / "manifests" / "checkpoint.yaml")

    environment = collect_environment(project_root)
    atomic_write_json(run_dir / "environment.json", environment)

    record = {
        "schema_version": 1,
        "run_id": identifier,
        "status": "pilot",
        "started_at": utc_now(),
        "completed_at": None,
        "config_path": str(config_path) if config_path else None,
        "config_sha256": digest,
        "git_commit": environment["git"]["commit"],
        "dataset": config["data"]["dataset"],
        "dataset_split": config["data"]["split"],
        "dataset_manifest_path": str(config["data"]["manifest"]),
        "dataset_manifest_sha256": file_digest(dataset_manifest_path),
        "dataset_verification": dataset_verification,
        "model": config["model"]["name"],
        "pretrained": config["model"]["pretrained"],
        "checkpoint": config["model"].get("checkpoint"),
        "checkpoint_manifest_path": str(config["model"]["manifest"]),
        "checkpoint_manifest_sha256": file_digest(checkpoint_manifest_path),
        "checkpoint_verification": checkpoint_verification,
        "seed": config["experiment"]["seed"],
        "precision": config["model"].get("precision"),
        "metrics_file": None,
        "notes": None,
        "review": None,
    }
    validate_payload("run", record)
    atomic_write_json(run_dir / "run.json", record)
    return run_dir


def finalize_run(
    run_dir: str | Path,
    metrics: dict[str, int | float],
    *,
    status: str = "complete",
    notes: str | None = None,
) -> dict[str, Any]:
    """Validate metrics and atomically finalize a non-reportable run."""
    if status not in FINALIZABLE_STATUSES:
        if status == "reportable":
            raise ValueError("reportable runs must be created with promote-run after review")
        raise ValueError(f"unsupported run status: {status}")
    normalized_metrics = _validate_metrics(metrics)
    directory = Path(run_dir)
    record = load_run(directory)

    atomic_write_json(directory / "metrics.json", normalized_metrics)
    record["status"] = status
    record["completed_at"] = utc_now()
    record["metrics_file"] = "metrics.json"
    record["notes"] = notes
    record["review"] = None
    validate_payload("run", record)
    atomic_write_json(directory / "run.json", record)
    return record


def fail_run(
    run_dir: str | Path,
    error: BaseException | str,
    *,
    notes: str | None = None,
) -> dict[str, Any]:
    """Retain a failed run without inventing numeric metrics."""
    directory = Path(run_dir)
    record = load_run(directory)
    error_text = str(error).strip() or type(error).__name__
    record["status"] = "failed"
    record["completed_at"] = utc_now()
    record["metrics_file"] = None
    record["notes"] = f"{notes + ': ' if notes else ''}{error_text}"
    record["review"] = None
    validate_payload("run", record)
    atomic_write_json(directory / "run.json", record)
    return record


def promote_run(
    run_dir: str | Path,
    *,
    reviewed_by: str,
    verification_run: str | Path | None = None,
    prediction_review: str | Path | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Promote one reviewed, clean, fully verified run to ``reportable``."""
    reviewer = reviewed_by.strip()
    if not reviewer:
        raise ValueError("reviewed_by must be a non-empty string")

    directory = Path(run_dir)
    record = load_run(directory)
    if record["status"] != "complete":
        raise ValueError("only complete runs can be promoted to reportable")
    if record["review"] is not None:
        raise ValueError("complete promotion candidates must have null review metadata")

    if record["dataset"] == "winoground":
        canonical_integrity = validate_completed_winoground_run_integrity(
            directory,
            expected_cache_enabled=True,
        )
        if verification_run is None:
            raise ValueError("Winoground promotion requires a cache-disabled verification run")
        if prediction_review is None:
            raise ValueError("Winoground promotion requires prediction review evidence")
        verification_directory = Path(verification_run)
        if _same_directory(directory, verification_directory):
            raise ValueError("verification run must be different from canonical run")
        verification_integrity = validate_completed_winoground_run_integrity(
            verification_directory,
            expected_cache_enabled=False,
        )
        verification_record = verification_integrity["run"]
        if verification_record["run_id"] == record["run_id"]:
            raise ValueError("verification run ID must differ from canonical run ID")

        comparison = compare_runs(directory, verification_directory)
        if comparison.get("passed") is not True:
            gates = ", ".join(_failed_comparison_gates(comparison)[:5])
            suffix = f": {gates}" if gates else ""
            raise ValueError(f"cache-disabled verification comparison did not pass{suffix}")
        review_summary = validate_prediction_review(
            prediction_review,
            canonical_integrity["prediction_sample_ids"],
        )
        reviewed_at = utc_now()
        evidence_fields, installed_review = _install_promotion_evidence(
            directory,
            comparison=comparison,
            prediction_review=Path(prediction_review),
            review_summary=review_summary,
            audit=canonical_integrity["audit"],
            canonical_integrity=canonical_integrity,
            verification_integrity=verification_integrity,
            reviewed_by=reviewer,
            reviewed_at=reviewed_at,
            notes=notes,
        )
        record["dataset_verification"] = canonical_integrity["dataset_verification"]
        record["checkpoint_verification"] = canonical_integrity["checkpoint_verification"]
        record["status"] = "reportable"
        record["review"] = {
            "reviewed_by": reviewer,
            "reviewed_at": reviewed_at,
            "decision": "reportable",
            "notes": notes,
            **evidence_fields,
        }
        validate_payload("run", record)
        try:
            atomic_write_json(directory / "run.json", record)
        except Exception:
            shutil.rmtree(installed_review, ignore_errors=True)
            raise
        return record

    environment = _load_json(directory / "environment.json", "environment")

    git = environment["git"]
    if git["commit"] is None or record["git_commit"] != git["commit"]:
        raise ValueError("run Git commit is missing or inconsistent with environment.json")
    if git["dirty"] is not False:
        raise ValueError("dirty or unknown Git state cannot be promoted to reportable")

    config = _load_yaml(directory / "config.resolved.yaml")
    if config_digest(config) != record["config_sha256"]:
        raise ValueError("run config digest does not match resolved config")
    _validate_run_config_identity(record, config)
    project_root = repository_root()
    dataset_manifest_path = directory / "manifests" / "dataset.yaml"
    dataset_manifest = load_dataset_manifest(dataset_manifest_path)
    if file_digest(dataset_manifest_path) != record["dataset_manifest_sha256"]:
        raise ValueError("run dataset manifest digest does not match snapshotted manifest")
    _validate_run_dataset_identity(record, config, dataset_manifest)
    dataset_root = _resolve_path(config["data"]["root"], project_root)
    _validate_reportable_image_inventory(record["dataset"], dataset_root, dataset_manifest)
    dataset_verification = _verification_record(dataset_root, dataset_manifest)
    if dataset_verification["status"] != "passed" or dataset_verification["declared_files"] <= 0:
        raise ValueError("dataset files must be declared and verified before promotion")
    _validate_reportable_dataset_provenance(record["dataset"], dataset_manifest)
    checkpoint_manifest_path = directory / "manifests" / "checkpoint.yaml"
    checkpoint_manifest = load_checkpoint_manifest(checkpoint_manifest_path)
    if file_digest(checkpoint_manifest_path) != record["checkpoint_manifest_sha256"]:
        raise ValueError("run checkpoint manifest digest does not match snapshotted manifest")
    checkpoint_root = _resolve_path(config["model"].get("checkpoint_root", "."), project_root)
    checkpoint_verification = _verification_record(checkpoint_root, checkpoint_manifest)
    if checkpoint_verification["status"] == "failed":
        raise ValueError("checkpoint manifest file verification failed")

    metrics_path = directory / str(record.get("metrics_file") or "metrics.json")
    metrics = _load_json(metrics_path, "metrics")
    _validate_metrics(metrics)

    reviewed_at = utc_now()
    review_record: dict[str, Any] = {
        "reviewed_by": reviewer,
        "reviewed_at": reviewed_at,
        "decision": "reportable",
        "notes": notes,
    }
    record["dataset_verification"] = dataset_verification
    record["checkpoint_verification"] = checkpoint_verification
    record["status"] = "reportable"
    record["review"] = review_record
    validate_payload("run", record)
    atomic_write_json(directory / "run.json", record)
    return record


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Load and validate an existing run record."""
    path = Path(run_dir) / "run.json"
    return _load_json(path, "run")


def validate_completed_winoground_run_integrity(
    run_dir: str | Path,
    *,
    expected_cache_enabled: bool,
    require_clean_git: bool = True,
) -> dict[str, Any]:
    """Fully validate one immutable, completed Winoground run."""
    if not isinstance(expected_cache_enabled, bool):
        raise ValueError("expected_cache_enabled must be a boolean")
    directory = Path(run_dir)
    run = load_run(directory)
    if run["status"] != "complete":
        raise ValueError("Winoground integrity validation requires status complete")
    if run["review"] is not None:
        raise ValueError("Winoground integrity validation requires review null")
    if run["dataset"] != "winoground":
        raise ValueError("Winoground integrity validation requires dataset winoground")

    config = _load_yaml(directory / "config.resolved.yaml")
    if config_digest(config) != run["config_sha256"]:
        raise ValueError("run config digest does not match resolved config")

    environment = _load_json(directory / "environment.json", "environment")
    git = environment["git"]
    if require_clean_git and git.get("dirty") is not False:
        raise ValueError("dirty or unknown Git state cannot be used for reportable results")
    if require_clean_git and git.get("untracked_count", 0) != 0:
        raise ValueError("untracked files cannot be used for reportable results")

    project_root = repository_root()
    dataset_manifest_path = directory / "manifests" / "dataset.yaml"
    dataset_manifest = load_dataset_manifest(dataset_manifest_path)
    if file_digest(dataset_manifest_path) != run["dataset_manifest_sha256"]:
        raise ValueError("run dataset manifest digest does not match snapshotted manifest")
    if dataset_manifest.get("splits", {}).get("test") != 400:
        raise ValueError("Winoground reportable runs require exactly 400 test samples")
    dataset_root = _resolve_path(config["data"]["root"], project_root)
    _validate_reportable_image_inventory(run["dataset"], dataset_root, dataset_manifest)
    dataset_failures = verify_manifest_files(
        dataset_root,
        dataset_manifest,
        require_hashed_inventory=True,
    )
    if dataset_failures:
        raise ValueError(f"dataset file verification failed: {dataset_failures[0]}")
    dataset_verification = _verification_record(dataset_root, dataset_manifest)
    if dataset_verification["status"] != "passed" or dataset_verification["declared_files"] <= 0:
        raise ValueError("dataset files must be declared and verified before promotion")
    _validate_reportable_dataset_provenance(run["dataset"], dataset_manifest)

    evaluation = _load_json(directory / "evaluation.json", "evaluation")
    validate_winoground_run_identity(
        run,
        config=config,
        environment=environment,
        dataset_manifest=dataset_manifest,
        evaluation=evaluation,
    )
    metadata = evaluation["metadata"]
    if metadata.get("benchmark") != "winoground_paired_matrix":
        raise ValueError("evaluation benchmark must be winoground_paired_matrix")
    if metadata.get("num_samples") != 400:
        raise ValueError("Winoground evaluation must contain exactly 400 samples")
    if evaluation.get("predictions_file") != "predictions.jsonl":
        raise ValueError("evaluation must reference predictions.jsonl")
    if run.get("metrics_file") != "metrics.json":
        raise ValueError("run must reference metrics.json")
    metrics = _load_json(directory / "metrics.json", "metrics")
    if evaluation.get("metrics") != metrics:
        raise ValueError("metrics.json and evaluation.json metrics differ")
    _validate_metrics(metrics)

    _validate_winoground_annotation_path(config, dataset_root, project_root)
    annotation_path = dataset_root / "annotations" / f"{run['dataset_split']}.jsonl"
    coverage_failures = verify_annotation_inventory_coverage(
        dataset_root,
        dataset_manifest,
        split=run["dataset_split"],
        expected_annotation_sha256=metadata.get("annotation_sha256", ""),
    )
    if coverage_failures:
        raise ValueError(
            f"annotation and image inventory verification failed: {coverage_failures[0]}"
        )

    checkpoint_manifest_path = directory / "manifests" / "checkpoint.yaml"
    checkpoint_manifest = load_checkpoint_manifest(checkpoint_manifest_path)
    if file_digest(checkpoint_manifest_path) != run["checkpoint_manifest_sha256"]:
        raise ValueError("run checkpoint manifest digest does not match snapshotted manifest")
    checkpoint_root = _resolve_path(config["model"].get("checkpoint_root", "."), project_root)
    checkpoint_verification = _verification_record(checkpoint_root, checkpoint_manifest)
    if checkpoint_manifest.get("files") and checkpoint_verification["status"] != "passed":
        raise ValueError("checkpoint manifest file verification failed")
    if checkpoint_verification["status"] == "failed":
        raise ValueError("checkpoint manifest file verification failed")

    cache = metadata.get("cache")
    if not isinstance(cache, dict) or any(
        field not in cache or not isinstance(cache[field], bool)
        for field in ("enabled", "images_hit", "texts_hit")
    ):
        raise ValueError("evaluation cache metadata must contain boolean fields")
    if cache["enabled"] is not expected_cache_enabled:
        state = "enabled" if expected_cache_enabled else "disabled"
        raise ValueError(f"Winoground run cache must be {state}")
    if not expected_cache_enabled and (
        cache["images_hit"] is not False or cache["texts_hit"] is not False
    ):
        raise ValueError("cache-disabled verification run must report no cache hits")

    audit = audit_winoground_run(
        directory,
        expected_samples=400,
        annotation_path=annotation_path,
        require_annotation_alignment=True,
        write_outputs=False,
    )
    prediction_sample_ids = load_prediction_sample_ids(directory)
    artifact_digests = {
        relative: sha256_file(directory / relative) for relative in WINOGROUND_ARTIFACTS
    }
    return {
        "directory": directory,
        "run": run,
        "config": config,
        "environment": environment,
        "dataset_manifest": dataset_manifest,
        "checkpoint_manifest": checkpoint_manifest,
        "dataset_verification": dataset_verification,
        "checkpoint_verification": checkpoint_verification,
        "evaluation": evaluation,
        "metrics": metrics,
        "audit": audit,
        "prediction_sample_ids": prediction_sample_ids,
        "artifact_digests": artifact_digests,
    }


def validate_winoground_run_identity(
    run: dict[str, Any],
    *,
    config: dict[str, Any],
    environment: dict[str, Any],
    dataset_manifest: dict[str, Any],
    evaluation: dict[str, Any],
) -> None:
    """Bind mutable run metadata to immutable Winoground artifacts."""
    if run.get("dataset") != "winoground":
        raise ValueError("run dataset does not match reportable Winoground evidence")
    if config_digest(config) != run["config_sha256"]:
        raise ValueError("run config digest does not match resolved config")
    _validate_run_config_identity(run, config)
    _validate_run_dataset_identity(run, config, dataset_manifest)
    git = environment["git"]
    if not isinstance(git.get("commit"), str) or not git["commit"].strip():
        raise ValueError("environment Git commit must be non-empty")
    if run["git_commit"] != git["commit"]:
        raise ValueError("run Git commit is inconsistent with environment.json")
    if run.get("metrics_file") != "metrics.json":
        raise ValueError("run must reference metrics.json")
    _validate_evaluation_identity(run, evaluation, config=config)


def _validate_metrics(metrics: dict[str, int | float]) -> dict[str, float]:
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("metrics must be a non-empty mapping")
    normalized: dict[str, float] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("metric names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"metric {name!r} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"metric {name!r} must be finite")
        normalized[name] = numeric
    validate_payload("metrics", normalized)
    return normalized


def _verification_record(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    declared_files = len(manifest.get("files", []))
    if declared_files == 0:
        return {"status": "not_run", "declared_files": 0, "failures": []}
    failures = verify_manifest_files(root, manifest)
    return {
        "status": "failed" if failures else "passed",
        "declared_files": declared_files,
        "failures": failures,
    }


def _validate_dataset_identity(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    configured_dataset = config["data"]["dataset"]
    manifest_dataset = manifest["name"]
    if configured_dataset != manifest_dataset:
        raise ValueError(
            "dataset identity mismatch: "
            f"config declares {configured_dataset} but manifest declares {manifest_dataset}"
        )
    configured_split = config["data"]["split"]
    if configured_split not in manifest["splits"]:
        raise ValueError(
            "dataset split mismatch: "
            f"split {configured_split} is not declared by manifest {manifest_dataset}"
        )


def _validate_run_dataset_identity(
    record: dict[str, Any],
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    configured_dataset = config["data"]["dataset"]
    if record["dataset"] != configured_dataset:
        raise ValueError("run dataset does not match resolved config")
    if manifest["name"] != record["dataset"]:
        raise ValueError("dataset manifest name does not match run dataset")

    configured_split = config["data"]["split"]
    if record["dataset_split"] != configured_split:
        raise ValueError("run dataset split does not match resolved config")
    if record["dataset_split"] not in manifest["splits"]:
        raise ValueError("dataset manifest does not declare run split")


def _validate_run_config_identity(record: dict[str, Any], config: dict[str, Any]) -> None:
    model = config["model"]
    experiment = config["experiment"]
    checks = (
        ("model", model["name"], "run model does not match resolved config"),
        (
            "pretrained",
            model["pretrained"],
            "run pretrained weights do not match resolved config",
        ),
        ("seed", experiment["seed"], "run seed does not match resolved config"),
        ("precision", model.get("precision"), "run precision does not match resolved config"),
        ("checkpoint", model.get("checkpoint"), "run checkpoint does not match resolved config"),
    )
    for field, configured, error in checks:
        if record.get(field) != configured:
            raise ValueError(error)


def _validate_evaluation_identity(
    record: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
) -> None:
    metadata = evaluation["metadata"]
    if metadata.get("dataset") != record["dataset"]:
        raise ValueError("evaluation dataset does not match run")
    if metadata.get("split") != record["dataset_split"]:
        raise ValueError("evaluation split does not match run")
    if metadata.get("dataset_manifest_sha256") != record["dataset_manifest_sha256"]:
        raise ValueError("evaluation dataset manifest digest does not match run")
    encoder = metadata.get("encoder")
    if not isinstance(encoder, dict) or encoder.get("model") != record["model"]:
        raise ValueError("evaluation encoder model does not match run")
    if encoder.get("pretrained") != record["pretrained"]:
        raise ValueError("evaluation encoder pretrained weights do not match run")
    if encoder.get("precision") != record["precision"]:
        raise ValueError("evaluation encoder precision does not match run")
    if config is not None and encoder.get("framework") != config["model"]["framework"]:
        raise ValueError("evaluation encoder framework does not match resolved config")


def _validate_winoground_annotation_path(
    config: dict[str, Any],
    dataset_root: Path,
    project_root: Path,
) -> None:
    annotation_value = config["data"].get("annotation_file")
    if not isinstance(annotation_value, str) or not annotation_value.strip():
        raise ValueError("resolved config does not declare the Winoground evaluation annotation")
    annotation_path = _resolve_path(annotation_value, project_root)
    expected_path = dataset_root / "annotations" / f"{config['data']['split']}.jsonl"
    if annotation_path.resolve() != expected_path.resolve():
        raise ValueError("resolved config does not reference the normalized Winoground annotation")


def _same_directory(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return os.path.abspath(left) == os.path.abspath(right)


def _failed_comparison_gates(comparison: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    failures.extend(
        f"run identity {field}"
        for field, passed in comparison.get("run_identity_matches", {}).items()
        if passed is not True
    )
    failures.extend(
        f"evaluation metadata {field}"
        for field, passed in comparison.get("metadata_identity_matches", {}).items()
        if passed is not True
    )
    scalar_gates = {
        "cached_run_cache_enabled": "canonical cache enabled",
        "no_cache_run_cache_disabled": "verification cache disabled",
        "cached_run_status_valid": "canonical status",
        "no_cache_run_status_valid": "verification status",
        "metric_names_match": "metric names",
        "metrics_within_tolerance": "metrics tolerance",
        "prediction_count_match": "prediction count",
        "sample_id_order_match": "sample ID order",
        "scores_within_tolerance": "score tolerance",
    }
    failures.extend(
        label for field, label in scalar_gates.items() if comparison.get(field) is not True
    )
    if any(comparison.get("decision_differences", {}).values()):
        failures.append("prediction decisions")
    return failures


REVIEW_FIELDS = (
    "sample_id",
    "review_group",
    "mapping_checked",
    "visual_review_status",
    "annotation_issue",
    "notes",
)
VISUAL_REVIEW_STATUSES = {"pass", "issue", "uncertain"}
ANNOTATION_ISSUES = {"none", "possible", "confirmed"}


def validate_prediction_review(
    review_path: str | Path,
    prediction_ids: list[str],
) -> dict[str, Any]:
    path = Path(review_path)
    if not path.is_file():
        raise FileNotFoundError("prediction review CSV is missing")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing_fields = [field for field in REVIEW_FIELDS if field not in header]
        if missing_fields:
            raise ValueError(
                f"prediction review CSV is missing required headers: {', '.join(missing_fields)}"
            )
        rows = list(reader)

    expected_ids = set(prediction_ids)
    observed_ids: set[str] = set()
    visual_counts: Counter[str] = Counter()
    annotation_counts: Counter[str] = Counter()
    mapping_checked_count = 0
    for line_number, row in enumerate(rows, start=2):
        sample_id = (row.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError(f"prediction review line {line_number} has an empty sample_id")
        if sample_id in observed_ids:
            raise ValueError(f"prediction review contains duplicate sample_id: {sample_id}")
        observed_ids.add(sample_id)
        if not (row.get("review_group") or "").strip():
            raise ValueError(f"prediction review line {line_number} has an empty review_group")
        mapping_checked = (row.get("mapping_checked") or "").strip().lower()
        if mapping_checked not in {"true", "false"}:
            raise ValueError("prediction review mapping_checked must be true or false")
        if mapping_checked != "true":
            raise ValueError("every prediction review row must have mapping_checked=true")
        mapping_checked_count += 1
        visual_status = (row.get("visual_review_status") or "").strip().lower()
        if visual_status not in VISUAL_REVIEW_STATUSES:
            raise ValueError("prediction review visual_review_status is invalid")
        annotation_issue = (row.get("annotation_issue") or "").strip().lower()
        if annotation_issue not in ANNOTATION_ISSUES:
            raise ValueError("prediction review annotation_issue is invalid")
        notes = (row.get("notes") or "").strip()
        if visual_status in {"issue", "uncertain"} and not notes:
            raise ValueError(f"prediction review {visual_status} rows require notes")
        visual_counts[visual_status] += 1
        annotation_counts[annotation_issue] += 1

    missing_ids = sorted(expected_ids - observed_ids)
    extra_ids = sorted(observed_ids - expected_ids)
    if missing_ids:
        raise ValueError(f"prediction review is missing {len(missing_ids)} prediction sample IDs")
    if extra_ids:
        raise ValueError(f"prediction review contains {len(extra_ids)} unknown sample IDs")
    if len(rows) != len(prediction_ids):
        raise ValueError("prediction review row count does not match predictions")
    return {
        "prediction_review_count": len(rows),
        "mapping_checked_count": mapping_checked_count,
        "visual_review_status_counts": {
            status: visual_counts[status] for status in sorted(VISUAL_REVIEW_STATUSES)
        },
        "annotation_issue_counts": {
            status: annotation_counts[status] for status in sorted(ANNOTATION_ISSUES)
        },
    }


_validate_prediction_review = validate_prediction_review


def _install_promotion_evidence(
    directory: Path,
    *,
    comparison: dict[str, Any],
    prediction_review: Path,
    review_summary: dict[str, Any],
    audit: dict[str, Any],
    canonical_integrity: dict[str, Any],
    verification_integrity: dict[str, Any],
    reviewed_by: str,
    reviewed_at: str,
    notes: str | None,
) -> tuple[dict[str, Any], Path]:
    review_dir = directory / "review"
    if review_dir.exists():
        raise ValueError("promotion evidence already exists")
    staging = Path(tempfile.mkdtemp(prefix=".review-staging-", dir=directory))
    try:
        comparison_path = staging / "run_comparison.json"
        atomic_write_json(comparison_path, comparison)
        comparison_sha = sha256_file(comparison_path)
        review_path = staging / "prediction_review.csv"
        shutil.copyfile(prediction_review, review_path)
        review_sha = sha256_file(review_path)
        canonical_run = canonical_integrity["run"]
        verification_run = verification_integrity["run"]
        evidence = {
            "schema_version": 1,
            "canonical_run_id": canonical_run["run_id"],
            "verification_run_id": verification_run["run_id"],
            "canonical_config_sha256": canonical_run["config_sha256"],
            "canonical_dataset_manifest_sha256": canonical_run["dataset_manifest_sha256"],
            "canonical_checkpoint_manifest_sha256": canonical_run[
                "checkpoint_manifest_sha256"
            ],
            "canonical_artifacts": canonical_integrity["artifact_digests"],
            "verification_artifacts": verification_integrity["artifact_digests"],
            "verification_run_record_sha256": sha256_file(
                verification_integrity["directory"] / "run.json"
            ),
            "comparison_file": "review/run_comparison.json",
            "comparison_sha256": comparison_sha,
            "comparison_passed": True,
            "prediction_review_file": "review/prediction_review.csv",
            "prediction_review_sha256": review_sha,
            **review_summary,
            "prediction_audit": {
                "prediction_count": audit["prediction_count"],
                "unique_sample_ids": audit["unique_sample_ids"],
                "metrics_recomputed": audit["metrics_recomputed"],
                "decisions_recomputed": audit["decisions_recomputed"],
                "annotation_alignment_verified": audit["annotation_alignment_verified"],
                "all_recomputed_metrics_verified": audit[
                    "all_recomputed_metrics_verified"
                ],
                "recomputed_metric_count": audit["recomputed_metric_count"],
            },
            "reviewed_by": reviewed_by,
            "reviewed_at": reviewed_at,
            "notes": notes,
        }
        evidence_path = staging / "promotion_evidence.json"
        atomic_write_json(evidence_path, evidence)
        evidence_sha = sha256_file(evidence_path)
        os.replace(staging, review_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return (
        {
            "verification_run_id": verification_run["run_id"],
            "promotion_evidence_file": "review/promotion_evidence.json",
            "promotion_evidence_sha256": evidence_sha,
            "comparison_file": "review/run_comparison.json",
            "comparison_sha256": comparison_sha,
            "prediction_review_file": "review/prediction_review.csv",
            "prediction_review_sha256": review_sha,
            "prediction_review_count": review_summary["prediction_review_count"],
        },
        review_dir,
    )


def _validate_reportable_image_inventory(
    dataset_name: str,
    dataset_root: Path,
    manifest: dict[str, Any],
) -> None:
    processing = manifest.get("processing")
    has_inventory = isinstance(processing, dict) and "image_inventory" in processing
    if dataset_name != "winoground" and not has_inventory:
        return
    failures = verify_hashed_image_inventory(dataset_root, manifest)
    if failures:
        preview = "; ".join(failures[:5])
        raise ValueError(f"hashed image inventory verification failed: {preview}")


def _validate_reportable_dataset_provenance(
    dataset_name: str,
    manifest: dict[str, Any],
) -> None:
    """Enforce dataset-specific provenance required for reportable promotion."""
    if dataset_name != "winoground":
        return

    processing = manifest.get("processing")
    if not isinstance(processing, dict):
        raise ValueError("Winoground dataset provenance is not pinned and verified")
    if processing.get("provenance_status") != "pinned_revision_verified":
        raise ValueError("Winoground dataset provenance is not pinned and verified")
    if processing.get("source_dataset") != "facebook/winoground":
        raise ValueError("Winoground source_dataset must be facebook/winoground")
    if processing.get("source_split") != "test":
        raise ValueError("Winoground source_split must be test")

    revision = processing.get("source_revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        raise ValueError("Winoground source_revision must be a 40-character commit SHA")
    exporter_version = processing.get("exporter_version")
    if not isinstance(exporter_version, str) or not exporter_version.strip():
        raise ValueError("Winoground exporter_version is missing")
    if not _is_rfc3339_utc(manifest.get("downloaded_at")):
        raise ValueError("Winoground downloaded_at is missing or invalid")
    if manifest.get("splits", {}).get("test") != 400:
        raise ValueError("Winoground reportable runs require exactly 400 test samples")
    if processing.get("image_hashes") is not True:
        raise ValueError("Winoground reportable runs require hashed images")

    match_rate = processing.get("caption_alphanumeric_character_multiset_match_rate")
    method = processing.get("caption_alphanumeric_character_multiset_method")
    if (
        isinstance(match_rate, bool)
        or not isinstance(match_rate, (int, float))
        or float(match_rate) != 100.0
        or method != WINOGROUND_CONTENT_CHECK_METHOD
    ):
        raise ValueError("Winoground caption content check is missing or invalid")


def _is_rfc3339_utc(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _load_json(path: Path, schema_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"record not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{schema_name} record must be a JSON object")
    validate_payload(schema_name, payload)
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"record not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML record must be an object: {path}")
    validate_config(payload)
    return payload


def _safe_component(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "-" for character in value
    )
    return cleaned.strip("-") or "run"
