#!/usr/bin/env python3
"""Compare a canonical baseline run with an independent no-cache rerun."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from recoalign.schema_validation import validate_payload

RUN_IDENTITY_FIELDS = (
    "dataset_manifest_sha256",
    "checkpoint_manifest_sha256",
    "dataset",
    "dataset_split",
    "model",
    "pretrained",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare cached and no-cache run outputs")
    parser.add_argument("cached_run")
    parser.add_argument("no_cache_run")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    return parser.parse_args()


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
    cached = _load_run(Path(cached_run))
    no_cache = _load_run(Path(no_cache_run))

    run_identity_matches = {
        field: cached["run"].get(field) == no_cache["run"].get(field)
        for field in RUN_IDENTITY_FIELDS
    }
    metadata_identity_matches = {
        field: cached["evaluation"]["metadata"].get(field)
        == no_cache["evaluation"]["metadata"].get(field)
        for field in METADATA_IDENTITY_FIELDS
    }

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

    no_cache_metadata = no_cache["evaluation"]["metadata"].get("cache")
    no_cache_has_no_hits = isinstance(no_cache_metadata, dict) and all(
        no_cache_metadata.get(field) is False for field in ("images_hit", "texts_hit")
    )
    summary = {
        "cached_run_id": cached["run"]["run_id"],
        "no_cache_run_id": no_cache["run"]["run_id"],
        "atol": atol,
        "rtol": rtol,
        "run_identity_matches": run_identity_matches,
        "metadata_identity_matches": metadata_identity_matches,
        "metric_names_match": metric_names_match,
        "metrics_within_tolerance": metrics_within_tolerance,
        "maximum_metric_absolute_difference": max(metric_differences.values(), default=0.0),
        "prediction_count_match": prediction_count_match,
        "sample_id_order_match": sample_id_order_match,
        "scores_within_tolerance": scores_within_tolerance,
        "maximum_score_absolute_difference": max_score_absolute_difference,
        "decision_differences": decision_differences,
        "no_cache_run_has_no_hits": no_cache_has_no_hits,
    }
    summary["passed"] = all(run_identity_matches.values()) and all(
        metadata_identity_matches.values()
    ) and all(
        (
            metric_names_match,
            metrics_within_tolerance,
            prediction_count_match,
            sample_id_order_match,
            scores_within_tolerance,
            not any(decision_differences.values()),
            no_cache_has_no_hits,
        )
    )
    return summary


def _load_run(directory: Path) -> dict[str, Any]:
    run = _load_json(directory / "run.json")
    evaluation = _load_json(directory / "evaluation.json")
    metrics = _load_json(directory / "metrics.json")
    validate_payload("run", run)
    validate_payload("evaluation", evaluation)
    validate_payload("metrics", metrics)
    if run["status"] not in {"complete", "reportable"}:
        raise ValueError(f"run is not complete: {directory}")
    if evaluation["metrics"] != metrics:
        raise ValueError(f"metrics.json and evaluation.json differ: {directory}")
    predictions_file = evaluation.get("predictions_file")
    if predictions_file != "predictions.jsonl":
        raise ValueError(f"run must reference predictions.jsonl: {directory}")
    predictions = _load_jsonl(directory / predictions_file)
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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required run file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"required run file is missing: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"predictions must be JSON objects: {path}")
    return rows


def main() -> int:
    args = parse_args()
    try:
        summary = compare_runs(
            args.cached_run,
            args.no_cache_run,
            atol=args.atol,
            rtol=args.rtol,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
