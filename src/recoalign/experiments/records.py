"""Create, finalize, fail, and promote self-contained experiment records."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from recoalign.config import config_digest, file_digest, validate_config
from recoalign.data.manifest import (
    load_checkpoint_manifest,
    load_dataset_manifest,
    snapshot_manifest,
    verify_manifest_files,
)
from recoalign.reproducibility import atomic_write_json, collect_environment, utc_now
from recoalign.schema_validation import repository_root, validate_payload

RUN_STATUSES = {"pilot", "partial", "failed", "complete", "reportable"}
FINALIZABLE_STATUSES = RUN_STATUSES - {"reportable"}


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
    run_dir.mkdir(parents=True)

    project_root = repository_root()
    dataset_manifest_path = _resolve_path(config["data"]["manifest"], project_root)
    checkpoint_manifest_path = _resolve_path(config["model"]["manifest"], project_root)
    dataset_manifest = load_dataset_manifest(dataset_manifest_path)
    checkpoint_manifest = load_checkpoint_manifest(checkpoint_manifest_path)

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

    environment = _load_json(directory / "environment.json", "environment")
    metrics_path = directory / str(record.get("metrics_file") or "metrics.json")
    metrics = _load_json(metrics_path, "metrics")
    _validate_metrics(metrics)

    git = environment["git"]
    if git["commit"] is None or record["git_commit"] != git["commit"]:
        raise ValueError("run Git commit is missing or inconsistent with environment.json")
    if git["dirty"] is not False:
        raise ValueError("dirty or unknown Git state cannot be promoted to reportable")

    config = _load_yaml(directory / "config.resolved.yaml")
    project_root = repository_root()
    dataset_manifest = load_dataset_manifest(directory / "manifests" / "dataset.yaml")
    dataset_root = _resolve_path(config["data"]["root"], project_root)
    dataset_verification = _verification_record(dataset_root, dataset_manifest)
    if dataset_verification["status"] != "passed" or dataset_verification["declared_files"] <= 0:
        raise ValueError("dataset files must be declared and verified before promotion")
    processing = dataset_manifest.get("processing") or {}
    if processing.get("image_inventory") and processing.get("image_hashes") is not True:
        raise ValueError(
            "reportable image benchmarks require a SHA-256 image inventory; "
            "rerun dataset preparation with --hash-images"
        )

    checkpoint_manifest = load_checkpoint_manifest(directory / "manifests" / "checkpoint.yaml")
    checkpoint_root = _resolve_path(config["model"].get("checkpoint_root", "."), project_root)
    checkpoint_verification = _verification_record(checkpoint_root, checkpoint_manifest)
    if checkpoint_verification["status"] == "failed":
        raise ValueError("checkpoint manifest file verification failed")

    record["dataset_verification"] = dataset_verification
    record["checkpoint_verification"] = checkpoint_verification
    record["status"] = "reportable"
    record["review"] = {
        "reviewed_by": reviewer,
        "reviewed_at": utc_now(),
        "decision": "reportable",
        "notes": notes,
    }
    validate_payload("run", record)
    atomic_write_json(directory / "run.json", record)
    return record


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Load and validate an existing run record."""
    path = Path(run_dir) / "run.json"
    return _load_json(path, "run")


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
