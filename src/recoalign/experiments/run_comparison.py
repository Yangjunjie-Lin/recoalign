"""Strict cached/no-cache experiment run comparison."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from recoalign.schema_validation import validate_payload

RUN_IDENTITY_FIELDS = (
    "config_sha256",
    "git_commit",
    "dataset_manifest_sha256",
    "checkpoint_manifest_sha256",
    "dataset",
    "dataset_split",
    "model",
    "pretrained",
    "seed",
    "precision",
)
METADATA_IDENTITY_FIELDS = (
    "protocol_version",
    "protocol",
    "dataset",
    "split",
    "annotation_sha256",
    "dataset_manifest_sha256",
    "encoder",
)
DECISION_FIELDS = (
    "image_to_text_correct",
    "text_to_image_correct",
    "group_correct",
    "tie",
)


def compare_runs(
    cached_run: str | Path,
    no_cache_run: str | Path,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> dict[str, Any]:
    """Return a strict machine-readable comparison without changing either run."""
    if atol < 0 or rtol < 0 or not math.isfinite(atol) or not math.isfinite(rtol):
        raise ValueError("atol and rtol must be finite and non-negative")
    cached = _load_run(Path(cached_run), "cached")
    no_cache = _load_run(Path(no_cache_run), "no-cache")

    run_identity_matches = {
        field: cached["run"].get(field) == no_cache["run"].get(field)
        for field in RUN_IDENTITY_FIELDS
    }
    metadata_identity_matches = {
        field: cached["evaluation"]["metadata"].get(field)
        == no_cache["evaluation"]["metadata"].get(field)
        for field in METADATA_IDENTITY_FIELDS
    }
    cached_run_status_valid = cached["run"]["status"] in {"complete", "reportable"}
    no_cache_run_status_valid = no_cache["run"]["status"] == "complete"

    cached_cache_metadata = cached["evaluation"]["metadata"].get("cache")
    no_cache_cache_metadata = no_cache["evaluation"]["metadata"].get("cache")
    cached_cache_valid = _cache_metadata_is_valid(cached_cache_metadata)
    no_cache_cache_valid = _cache_metadata_is_valid(no_cache_cache_metadata)
    cached_cache_enabled = cached_cache_valid and cached_cache_metadata["enabled"] is True
    no_cache_disabled = no_cache_cache_valid and all(
        no_cache_cache_metadata[field] is False for field in ("enabled", "images_hit", "texts_hit")
    )

    metric_names_match = set(cached["metrics"]) == set(no_cache["metrics"])
    metric_differences: dict[str, float] = {}
    metrics_within_tolerance = metric_names_match
    if metric_names_match:
        for name in sorted(cached["metrics"]):
            left = float(cached["metrics"][name])
            right = float(no_cache["metrics"][name])
            difference = abs(left - right)
            metric_differences[name] = difference
            metrics_within_tolerance &= math.isclose(left, right, abs_tol=atol, rel_tol=rtol)

    cached_predictions = cached["predictions"]
    no_cache_predictions = no_cache["predictions"]
    prediction_count_match = len(cached_predictions) == len(no_cache_predictions)
    sample_id_order_match = prediction_count_match and all(
        left.get("sample_id") == right.get("sample_id")
        for left, right in zip(cached_predictions, no_cache_predictions, strict=True)
    )

    decision_differences = {field: 0 for field in DECISION_FIELDS}
    max_score_absolute_difference = 0.0
    scores_within_tolerance = prediction_count_match
    if prediction_count_match:
        for index, (left, right) in enumerate(
            zip(cached_predictions, no_cache_predictions, strict=True)
        ):
            left_scores = _scores(left, "cached", index)
            right_scores = _scores(right, "no-cache", index)
            max_score_absolute_difference = max(
                max_score_absolute_difference,
                float(np.max(np.abs(left_scores - right_scores))),
            )
            scores_within_tolerance &= bool(
                np.allclose(left_scores, right_scores, atol=atol, rtol=rtol)
            )
            for field in DECISION_FIELDS:
                if left.get(field) is not right.get(field):
                    decision_differences[field] += 1

    summary = {
        "cached_run_id": cached["run"]["run_id"],
        "no_cache_run_id": no_cache["run"]["run_id"],
        "atol": atol,
        "rtol": rtol,
        "run_identity_matches": run_identity_matches,
        "metadata_identity_matches": metadata_identity_matches,
        "cached_run_cache_enabled": cached_cache_enabled,
        "no_cache_run_cache_disabled": no_cache_disabled,
        "cached_run_status_valid": cached_run_status_valid,
        "no_cache_run_status_valid": no_cache_run_status_valid,
        "metric_names_match": metric_names_match,
        "metrics_within_tolerance": metrics_within_tolerance,
        "maximum_metric_absolute_difference": max(metric_differences.values(), default=0.0),
        "prediction_count_match": prediction_count_match,
        "sample_id_order_match": sample_id_order_match,
        "scores_within_tolerance": scores_within_tolerance,
        "maximum_score_absolute_difference": max_score_absolute_difference,
        "decision_differences": decision_differences,
    }
    summary["passed"] = (
        all(run_identity_matches.values())
        and all(metadata_identity_matches.values())
        and all(
            (
                metric_names_match,
                metrics_within_tolerance,
                prediction_count_match,
                sample_id_order_match,
                scores_within_tolerance,
                not any(decision_differences.values()),
                cached_cache_enabled,
                no_cache_disabled,
                cached_run_status_valid,
                no_cache_run_status_valid,
            )
        )
    )
    return summary


def _cache_metadata_is_valid(value: Any) -> bool:
    return isinstance(value, dict) and all(
        field in value and isinstance(value[field], bool)
        for field in ("enabled", "images_hit", "texts_hit")
    )


def _load_run(directory: Path, label: str) -> dict[str, Any]:
    run = _load_json(directory / "run.json", label, "run.json")
    evaluation = _load_json(directory / "evaluation.json", label, "evaluation.json")
    metrics = _load_json(directory / "metrics.json", label, "metrics.json")
    validate_payload("run", run)
    validate_payload("evaluation", evaluation)
    validate_payload("metrics", metrics)
    if evaluation["metrics"] != metrics:
        raise ValueError(f"{label} metrics.json and evaluation.json differ")
    predictions_file = evaluation.get("predictions_file")
    if predictions_file != "predictions.jsonl":
        raise ValueError(f"{label} run must reference predictions.jsonl")
    predictions = _load_jsonl(directory / predictions_file, label)
    for row in predictions:
        validate_payload("prediction", row)
    return {
        "run": run,
        "evaluation": evaluation,
        "metrics": metrics,
        "predictions": predictions,
    }


def _scores(row: dict[str, Any], label: str, index: int) -> np.ndarray:
    values = row.get("scores")
    if (
        not isinstance(values, list)
        or len(values) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        )
    ):
        raise ValueError(f"{label} prediction {index} must contain four finite scores")
    return np.asarray(values, dtype=np.float64)


def _load_json(path: Path, label: str, filename: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} run is missing {filename}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} {filename} must be a JSON object")
    return payload


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} run is missing predictions.jsonl")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{label} prediction line {line_number} must be an object")
        rows.append(payload)
    return rows
