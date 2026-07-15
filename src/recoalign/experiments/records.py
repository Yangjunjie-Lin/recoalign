"""Create and finalize self-contained experiment records."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from recoalign.config import config_digest, validate_config
from recoalign.reproducibility import atomic_write_json, collect_environment, utc_now

RUN_STATUSES = {"pilot", "partial", "failed", "complete", "reportable"}


def create_run(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    output_root: str | Path | None = None,
    run_id: str | None = None,
) -> Path:
    """Create a run directory with resolved config, environment, and provenance."""
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

    with (run_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)

    environment = collect_environment(Path.cwd())
    atomic_write_json(run_dir / "environment.json", environment)

    record = {
        "schema_version": 1,
        "run_id": identifier,
        "status": "pilot",
        "started_at": utc_now(),
        "completed_at": None,
        "config_path": str(config_path) if config_path else None,
        "config_sha256": digest,
        "git_commit": environment.get("git_commit"),
        "dataset": config["data"]["dataset"],
        "dataset_split": config["data"]["split"],
        "model": config["model"]["name"],
        "pretrained": config["model"]["pretrained"],
        "seed": config["experiment"]["seed"],
        "precision": config["model"].get("precision"),
        "checkpoint": config["model"].get("checkpoint"),
        "metrics_file": None,
        "notes": None,
    }
    atomic_write_json(run_dir / "run.json", record)
    return run_dir


def finalize_run(
    run_dir: str | Path,
    metrics: dict[str, int | float],
    *,
    status: str = "complete",
    notes: str | None = None,
) -> dict[str, Any]:
    """Validate metrics and atomically finalize an experiment record."""
    if status not in RUN_STATUSES:
        raise ValueError(f"unsupported run status: {status}")
    normalized_metrics = _validate_metrics(metrics)
    directory = Path(run_dir)
    record = load_run(directory)

    atomic_write_json(directory / "metrics.json", normalized_metrics)
    record["status"] = status
    record["completed_at"] = utc_now()
    record["metrics_file"] = "metrics.json"
    record["notes"] = notes
    atomic_write_json(directory / "run.json", record)
    return record


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Load an existing run record."""
    path = Path(run_dir) / "run.json"
    if not path.is_file():
        raise FileNotFoundError(f"run record not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("run record must be a JSON object")
    return payload


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
    return normalized


def _safe_component(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in value
    )
    return cleaned.strip("-") or "run"
