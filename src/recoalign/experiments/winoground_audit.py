"""Winoground prediction, decision, and metric audit."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from recoalign.evaluation.diagnostics import (
    evaluate_paired_matrix_scores,
    summarize_paired_matrix,
)
from recoalign.schema_validation import validate_payload

CORE_METRICS = (
    "image_to_text_accuracy",
    "text_to_image_accuracy",
    "group_accuracy",
    "tie_rate",
    "mean_image_to_text_margin",
    "mean_text_to_image_margin",
)
AUDIT_FIELDS = (
    "sample_id",
    "category",
    "tags",
    "scores",
    "image_to_text_correct",
    "text_to_image_correct",
    "group_correct",
    "tie",
    "image_to_text_margin",
    "text_to_image_margin",
    "minimum_margin",
)


def audit_winoground_run(
    run_dir: str | Path,
    *,
    expected_samples: int = 400,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Validate predictions and recompute Winoground decisions and metrics."""
    if (
        not isinstance(expected_samples, int)
        or isinstance(expected_samples, bool)
        or expected_samples <= 0
    ):
        raise ValueError("expected_samples must be a positive integer")
    directory = Path(run_dir)
    run = _load_json(directory / "run.json", "run.json")
    evaluation = _load_json(directory / "evaluation.json", "evaluation.json")
    metrics = _load_json(directory / "metrics.json", "metrics.json")
    validate_payload("run", run)
    validate_payload("evaluation", evaluation)
    validate_payload("metrics", metrics)
    if run["status"] not in {"complete", "reportable"}:
        raise ValueError(f"run status must be complete or reportable, observed {run['status']!r}")
    metadata = evaluation["metadata"]
    if metadata.get("benchmark") != "winoground_paired_matrix":
        raise ValueError("evaluation benchmark must be winoground_paired_matrix")
    if metadata.get("num_samples") != expected_samples:
        raise ValueError(
            f"evaluation expected {expected_samples} samples, "
            f"observed {metadata.get('num_samples')}"
        )
    if evaluation.get("predictions_file") != "predictions.jsonl":
        raise ValueError("evaluation must reference predictions.jsonl")

    predictions = _load_jsonl(directory / "predictions.jsonl")
    if len(predictions) != expected_samples:
        raise ValueError(f"expected {expected_samples} predictions, observed {len(predictions)}")
    sample_ids = [row.get("sample_id") for row in predictions]
    if any(not isinstance(sample_id, str) or not sample_id.strip() for sample_id in sample_ids):
        raise ValueError("every prediction must have a non-empty sample_id")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("prediction sample IDs must be unique")

    score_rows: list[list[float]] = []
    categories: list[str] = []
    tags: list[list[str]] = []
    for index, row in enumerate(predictions):
        validate_payload("prediction", row)
        raw_scores = row.get("scores")
        if (
            not isinstance(raw_scores, list)
            or len(raw_scores) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in raw_scores
            )
        ):
            raise ValueError(f"prediction {index}: scores must contain four finite numbers")
        category = row.get("category")
        sample_tags = row.get("tags")
        if not isinstance(category, str) or not category:
            raise ValueError(f"prediction {index}: category must be non-empty")
        if not isinstance(sample_tags, list) or any(
            not isinstance(tag, str) or not tag for tag in sample_tags
        ):
            raise ValueError(f"prediction {index}: tags must be a list of non-empty strings")
        score_rows.append([float(value) for value in raw_scores])
        categories.append(category)
        tags.append(sample_tags)

    result = evaluate_paired_matrix_scores(np.asarray(score_rows).reshape(-1, 2, 2))
    recomputed = summarize_paired_matrix(result, categories, tags)
    _verify_decisions(predictions, result)
    _verify_metrics(metrics, evaluation["metrics"], recomputed)

    rows = _audit_rows(predictions, result)
    groups = {
        "correct_samples.csv": [row for row in rows if row["group_correct"]],
        "group_failures.csv": [row for row in rows if not row["group_correct"]],
        "i2t_only_failures.csv": [
            row for row in rows if not row["image_to_text_correct"] and row["text_to_image_correct"]
        ],
        "t2i_only_failures.csv": [
            row for row in rows if row["image_to_text_correct"] and not row["text_to_image_correct"]
        ],
        "tie_samples.csv": [row for row in rows if row["tie"]],
        "lowest_margin_samples.csv": sorted(rows, key=lambda row: row["minimum_margin"])[
            : min(20, len(rows))
        ],
    }
    summary = {
        "run_id": run["run_id"],
        "run_status": run["status"],
        "prediction_count": len(rows),
        "unique_sample_ids": len(set(sample_ids)),
        "group_correct_count": int(result.group_correct.sum()),
        "group_failure_count": int((~result.group_correct).sum()),
        "image_to_text_only_failure_count": len(groups["i2t_only_failures.csv"]),
        "text_to_image_only_failure_count": len(groups["t2i_only_failures.csv"]),
        "tie_count": int(result.ties.sum()),
        "recomputed_metrics": {name: recomputed[name] for name in CORE_METRICS},
        "metrics_recomputed": True,
        "decisions_recomputed": True,
        "semantic_review": "not_performed",
    }
    if write_outputs:
        output = directory / "audit"
        output.mkdir(parents=True, exist_ok=True)
        for name, group in groups.items():
            _write_csv(output / name, group)
        _atomic_write_json(output / "audit_summary.json", summary)
    return summary


def load_prediction_sample_ids(run_dir: str | Path) -> list[str]:
    """Load validated prediction sample IDs for review-evidence matching."""
    predictions = _load_jsonl(Path(run_dir) / "predictions.jsonl")
    for row in predictions:
        validate_payload("prediction", row)
    sample_ids = [row.get("sample_id") for row in predictions]
    if any(not isinstance(sample_id, str) or not sample_id.strip() for sample_id in sample_ids):
        raise ValueError("every prediction must have a non-empty sample_id")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("prediction sample IDs must be unique")
    return sample_ids


def _verify_decisions(predictions: list[dict[str, Any]], result: Any) -> None:
    fields = {
        "image_to_text_correct": result.image_to_text_correct,
        "text_to_image_correct": result.text_to_image_correct,
        "group_correct": result.group_correct,
        "tie": result.ties,
    }
    for field, observed in fields.items():
        for index, expected in enumerate(observed):
            if predictions[index].get(field) is not bool(expected):
                raise ValueError(f"prediction {index}: {field} does not match recomputed scores")


def _verify_metrics(
    metrics: dict[str, Any],
    evaluation_metrics: dict[str, Any],
    recomputed: dict[str, float],
) -> None:
    if metrics != evaluation_metrics:
        raise ValueError("metrics.json and evaluation.json metrics differ")
    for name in CORE_METRICS:
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"metric {name!r} is missing or non-numeric")
        if not math.isclose(float(value), recomputed[name], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"metric {name!r} does not match recomputed predictions")


def _audit_rows(predictions: list[dict[str, Any]], result: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        image_margin = float(result.image_to_text_margins[index])
        text_margin = float(result.text_to_image_margins[index])
        rows.append(
            {
                "sample_id": prediction["sample_id"],
                "category": prediction["category"],
                "tags": json.dumps(prediction["tags"], ensure_ascii=False),
                "scores": json.dumps(prediction["scores"]),
                "image_to_text_correct": bool(result.image_to_text_correct[index]),
                "text_to_image_correct": bool(result.text_to_image_correct[index]),
                "group_correct": bool(result.group_correct[index]),
                "tie": bool(result.ties[index]),
                "image_to_text_margin": image_margin,
                "text_to_image_margin": text_margin,
                "minimum_margin": min(image_margin, text_margin),
            }
        )
    return rows


def _load_json(path: Path, filename: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required run file is missing: {filename}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {filename}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError("required run file is missing: predictions.jsonl")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"prediction line {line_number} must be an object")
        rows.append(payload)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
