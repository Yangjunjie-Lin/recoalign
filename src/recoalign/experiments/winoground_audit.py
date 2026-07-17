"""Winoground prediction, decision, and metric audit."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from recoalign.benchmarks.caption_multisets import caption_multiset_matches
from recoalign.evaluation.diagnostics import (
    evaluate_paired_matrix_scores,
    summarize_paired_matrix,
)
from recoalign.schema_validation import validate_payload

CAPTION_CONTENT_CHECK_METHOD = "casefolded_alphanumeric_character_multiset_v1"
CANONICAL_CAPTION_METRIC = "caption_alphanumeric_character_multiset_match_rate"
DEPRECATED_CAPTION_METRIC = "caption_token_multiset_match_rate"
ALLOWED_NON_PAIRED_MATRIX_METRICS = {
    CANONICAL_CAPTION_METRIC,
    DEPRECATED_CAPTION_METRIC,
}
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
    annotation_path: str | Path | None = None,
    require_annotation_alignment: bool = False,
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

    annotation_rows: list[dict[str, Any]] | None = None
    caption_metric: float | None = None
    if annotation_path is not None:
        annotation_rows = _load_jsonl(Path(annotation_path), label="normalized annotation")
        caption_metric = _verify_annotation_alignment(
            predictions,
            annotation_rows,
            expected_samples=expected_samples,
        )
    elif require_annotation_alignment:
        raise ValueError("normalized annotation path is required for annotation alignment")

    result = evaluate_paired_matrix_scores(np.asarray(score_rows).reshape(-1, 2, 2))
    recomputed = summarize_paired_matrix(result, categories, tags)
    _verify_decisions(predictions, result)
    _verify_metrics(
        metrics,
        evaluation["metrics"],
        recomputed,
        caption_metric=caption_metric,
        require_caption_metric=require_annotation_alignment,
    )

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
        "recomputed_metrics": recomputed,
        "recomputed_metric_count": len(recomputed),
        "metrics_recomputed": True,
        "decisions_recomputed": True,
        "annotation_alignment_verified": annotation_rows is not None,
        "all_recomputed_metrics_verified": True,
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
    *,
    caption_metric: float | None,
    require_caption_metric: bool,
) -> None:
    if metrics != evaluation_metrics:
        raise ValueError("metrics.json and evaluation.json metrics differ")
    for name, expected in recomputed.items():
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"metric {name!r} is missing or non-numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"metric {name!r} must be finite")
        if not math.isclose(float(value), float(expected), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"metric {name!r} does not match recomputed predictions")

    extra_metrics = set(metrics) - set(recomputed)
    unexpected = sorted(extra_metrics - ALLOWED_NON_PAIRED_MATRIX_METRICS)
    if unexpected:
        raise ValueError(f"unexpected Winoground metric: {unexpected[0]}")
    if require_caption_metric and CANONICAL_CAPTION_METRIC not in metrics:
        raise ValueError(f"metric {CANONICAL_CAPTION_METRIC!r} is missing or non-numeric")
    if CANONICAL_CAPTION_METRIC in metrics:
        canonical = metrics[CANONICAL_CAPTION_METRIC]
        if (
            isinstance(canonical, bool)
            or not isinstance(canonical, (int, float))
            or not math.isfinite(float(canonical))
        ):
            raise ValueError(f"metric {CANONICAL_CAPTION_METRIC!r} is missing or non-numeric")
        if caption_metric is not None and not math.isclose(
            float(canonical),
            caption_metric,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"metric {CANONICAL_CAPTION_METRIC!r} does not match normalized annotation"
            )
    if DEPRECATED_CAPTION_METRIC in metrics:
        if CANONICAL_CAPTION_METRIC not in metrics:
            if require_caption_metric:
                raise ValueError(
                    f"deprecated metric {DEPRECATED_CAPTION_METRIC!r} requires canonical metric"
                )
            return
        alias = metrics[DEPRECATED_CAPTION_METRIC]
        if (
            isinstance(alias, bool)
            or not isinstance(alias, (int, float))
            or not math.isfinite(float(alias))
            or not math.isclose(
                float(alias),
                float(metrics[CANONICAL_CAPTION_METRIC]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"deprecated metric {DEPRECATED_CAPTION_METRIC!r} does not match canonical metric"
            )


def _verify_annotation_alignment(
    predictions: list[dict[str, Any]],
    annotation_rows: list[dict[str, Any]],
    *,
    expected_samples: int,
) -> float:
    if len(annotation_rows) != expected_samples:
        raise ValueError(
            f"expected {expected_samples} normalized annotations, observed {len(annotation_rows)}"
        )
    if len(predictions) != len(annotation_rows):
        raise ValueError("prediction and normalized annotation row counts differ")

    seen_ids: set[str] = set()
    caption_matches = 0
    for index, (prediction, annotation) in enumerate(
        zip(predictions, annotation_rows, strict=True)
    ):
        sample_id = annotation.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"normalized annotation {index} is missing sample_id")
        if sample_id in seen_ids:
            raise ValueError(f"normalized annotation contains duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)
        if prediction["sample_id"] != sample_id:
            raise ValueError(
                f"prediction {index} sample_id does not match normalized annotation"
            )

        category = annotation.get("category")
        if not isinstance(category, str) or not category:
            raise ValueError(f"normalized annotation {index} category must be non-empty")
        if prediction["category"] != category:
            raise ValueError(
                f"prediction {index} category does not match normalized annotation"
            )

        annotation_tags = annotation.get("tags")
        if not isinstance(annotation_tags, list) or any(
            not isinstance(tag, str) or not tag for tag in annotation_tags
        ):
            raise ValueError(
                f"normalized annotation {index} tags must be a list of non-empty strings"
            )
        if len(set(annotation_tags)) != len(annotation_tags):
            raise ValueError(f"normalized annotation {index} contains duplicate tags")
        if prediction["tags"] != annotation_tags:
            raise ValueError(f"prediction {index} tags do not match normalized annotation")

        caption_0 = annotation.get("caption_0")
        caption_1 = annotation.get("caption_1")
        if not isinstance(caption_0, str) or not isinstance(caption_1, str):
            raise ValueError(f"normalized annotation {index} captions must be strings")
        caption_matches += caption_multiset_matches(
            caption_0,
            caption_1,
            method=CAPTION_CONTENT_CHECK_METHOD,
        )
    return 100.0 * caption_matches / len(annotation_rows)


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


def _load_jsonl(path: Path, *, label: str = "prediction") -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"required {label} file is missing: {path.name}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{label} line {line_number} must be an object")
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
