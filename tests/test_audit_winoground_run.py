from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from recoalign.evaluation.diagnostics import (
    evaluate_paired_matrix_scores,
    summarize_paired_matrix,
)
from recoalign.experiments.winoground_audit import audit_winoground_run

AUDITOR = SimpleNamespace(audit_winoground_run=audit_winoground_run)


def test_audit_recomputes_decisions_metrics_and_groups(tmp_path: Path, research_config) -> None:
    run_dir = _run_fixture(tmp_path, research_config)

    summary = AUDITOR.audit_winoground_run(run_dir, expected_samples=3)

    assert summary["prediction_count"] == 3
    assert summary["unique_sample_ids"] == 3
    assert summary["group_correct_count"] == 1
    assert summary["group_failure_count"] == 2
    assert summary["image_to_text_only_failure_count"] == 1
    assert summary["text_to_image_only_failure_count"] == 0
    assert summary["tie_count"] == 1
    audit = run_dir / "audit"
    assert len((audit / "correct_samples.csv").read_text().splitlines()) == 2
    assert len((audit / "group_failures.csv").read_text().splitlines()) == 3
    assert len((audit / "tie_samples.csv").read_text().splitlines()) == 2
    assert (audit / "audit_summary.json").is_file()


def test_audit_rejects_prediction_decision_mismatch(tmp_path: Path, research_config) -> None:
    run_dir = _run_fixture(tmp_path, research_config)
    prediction_path = run_dir / "predictions.jsonl"
    rows = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    rows[0]["group_correct"] = False
    _write_jsonl(prediction_path, rows)

    try:
        AUDITOR.audit_winoground_run(run_dir, expected_samples=3)
    except ValueError as exc:
        assert "group_correct does not match" in str(exc)
    else:
        raise AssertionError("expected prediction mismatch to fail")


def test_audit_accepts_valid_400_row_synthetic_fixture(tmp_path: Path, research_config) -> None:
    run_dir = _run_fixture(tmp_path, research_config, sample_count=400)
    annotation_path = _write_normalized_annotations(run_dir)

    summary = audit_winoground_run(
        run_dir,
        expected_samples=400,
        annotation_path=annotation_path,
        require_annotation_alignment=True,
        write_outputs=False,
    )

    assert summary["prediction_count"] == 400
    assert summary["metrics_recomputed"] is True
    assert summary["decisions_recomputed"] is True
    assert summary["annotation_alignment_verified"] is True
    assert summary["all_recomputed_metrics_verified"] is True
    assert summary["recomputed_metric_count"] > 6


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("sample_id", "other", "prediction 1 sample_id"),
        ("category", "other", "prediction 1 category"),
        ("tags", ["other"], "prediction 1 tags"),
    ],
)
def test_audit_rejects_prediction_annotation_row_mismatch(
    tmp_path: Path,
    research_config,
    field: str,
    value: object,
    error: str,
) -> None:
    run_dir = _run_fixture(tmp_path, research_config)
    annotation_path = _write_normalized_annotations(run_dir)
    rows = _read_jsonl(annotation_path)
    rows[1][field] = value
    _write_jsonl(annotation_path, rows)

    with pytest.raises(ValueError, match=error):
        audit_winoground_run(
            run_dir,
            expected_samples=3,
            annotation_path=annotation_path,
            require_annotation_alignment=True,
            write_outputs=False,
        )


def test_audit_rejects_prediction_annotation_order_mismatch(
    tmp_path: Path, research_config
) -> None:
    run_dir = _run_fixture(tmp_path, research_config)
    annotation_path = _write_normalized_annotations(run_dir)
    rows = _read_jsonl(annotation_path)
    rows[0], rows[1] = rows[1], rows[0]
    _write_jsonl(annotation_path, rows)

    with pytest.raises(ValueError, match="prediction 0 sample_id"):
        audit_winoground_run(
            run_dir,
            expected_samples=3,
            annotation_path=annotation_path,
            require_annotation_alignment=True,
            write_outputs=False,
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("missing", "normalized annotation 1 is missing sample_id"),
        ("duplicate", "duplicate sample_id"),
    ],
)
def test_audit_rejects_invalid_annotation_sample_ids(
    tmp_path: Path,
    research_config,
    mutation: str,
    error: str,
) -> None:
    run_dir = _run_fixture(tmp_path, research_config)
    annotation_path = _write_normalized_annotations(run_dir)
    rows = _read_jsonl(annotation_path)
    if mutation == "missing":
        rows[1].pop("sample_id")
    else:
        rows[1]["sample_id"] = rows[0]["sample_id"]
    _write_jsonl(annotation_path, rows)

    with pytest.raises(ValueError, match=error):
        audit_winoground_run(
            run_dir,
            expected_samples=3,
            annotation_path=annotation_path,
            require_annotation_alignment=True,
            write_outputs=False,
        )


def test_audit_rejects_missing_predictions_file(tmp_path: Path, research_config) -> None:
    run_dir = _run_fixture(tmp_path, research_config)
    (run_dir / "predictions.jsonl").unlink()

    with pytest.raises(FileNotFoundError, match="predictions.jsonl"):
        audit_winoground_run(run_dir, expected_samples=3)


def test_audit_rejects_wrong_evaluation_predictions_file(tmp_path: Path, research_config) -> None:
    run_dir = _run_fixture(tmp_path, research_config)
    path = run_dir / "evaluation.json"
    evaluation = json.loads(path.read_text())
    evaluation["predictions_file"] = "other.jsonl"
    path.write_text(json.dumps(evaluation), encoding="utf-8")

    with pytest.raises(ValueError, match="reference predictions.jsonl"):
        audit_winoground_run(run_dir, expected_samples=3)


def test_audit_rejects_prediction_count_mismatch(tmp_path: Path, research_config) -> None:
    run_dir = _run_fixture(tmp_path, research_config)
    path = run_dir / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    _write_jsonl(path, rows[:-1])

    with pytest.raises(ValueError, match="expected 3 predictions, observed 2"):
        audit_winoground_run(run_dir, expected_samples=3)


def test_audit_rejects_duplicate_prediction_id(tmp_path: Path, research_config) -> None:
    run_dir = _run_fixture(tmp_path, research_config)
    path = run_dir / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["sample_id"] = rows[0]["sample_id"]
    _write_jsonl(path, rows)

    with pytest.raises(ValueError, match="sample IDs must be unique"):
        audit_winoground_run(run_dir, expected_samples=3)


def test_audit_rejects_schema_invalid_prediction(tmp_path: Path, research_config) -> None:
    run_dir = _run_fixture(tmp_path, research_config)
    path = run_dir / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0].pop("category")
    _write_jsonl(path, rows)

    with pytest.raises(ValueError):
        audit_winoground_run(run_dir, expected_samples=3)


def test_audit_rejects_nonfinite_score(tmp_path: Path, research_config) -> None:
    run_dir = _run_fixture(tmp_path, research_config)
    path = run_dir / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["scores"][0] = float("nan")
    _write_jsonl(path, rows)

    with pytest.raises(ValueError, match="four finite numbers"):
        audit_winoground_run(run_dir, expected_samples=3)


def test_audit_rejects_metrics_evaluation_mismatch(tmp_path: Path, research_config) -> None:
    run_dir = _run_fixture(tmp_path, research_config)
    path = run_dir / "evaluation.json"
    evaluation = json.loads(path.read_text())
    evaluation["metrics"]["group_accuracy"] = 0.0
    path.write_text(json.dumps(evaluation), encoding="utf-8")

    with pytest.raises(ValueError, match="metrics differ"):
        audit_winoground_run(run_dir, expected_samples=3)


def test_audit_rejects_metrics_prediction_mismatch(tmp_path: Path, research_config) -> None:
    run_dir = _run_fixture(tmp_path, research_config)
    metrics_path = run_dir / "metrics.json"
    evaluation_path = run_dir / "evaluation.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["group_accuracy"] = 0.0
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    evaluation = json.loads(evaluation_path.read_text())
    evaluation["metrics"] = metrics
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match recomputed predictions"):
        audit_winoground_run(run_dir, expected_samples=3)


@pytest.mark.parametrize(
    "metric",
    [
        "group_accuracy/winoground",
        "image_to_text_accuracy/winoground",
        "text_to_image_accuracy/winoground",
        "macro_category_group_accuracy",
        "group_accuracy/tag/synthetic",
        "caption_alphanumeric_character_multiset_match_rate",
    ],
)
def test_audit_rejects_tampered_recomputable_metric(
    tmp_path: Path,
    research_config,
    metric: str,
) -> None:
    run_dir = _run_fixture(tmp_path, research_config, sample_count=400)
    annotation_path = _write_normalized_annotations(run_dir)
    _mutate_metric(run_dir, metric, 0.0)

    with pytest.raises(ValueError, match="does not match"):
        audit_winoground_run(
            run_dir,
            expected_samples=400,
            annotation_path=annotation_path,
            require_annotation_alignment=True,
            write_outputs=False,
        )


def test_audit_rejects_deprecated_caption_alias_mismatch(
    tmp_path: Path, research_config
) -> None:
    run_dir = _run_fixture(tmp_path, research_config, sample_count=400)
    annotation_path = _write_normalized_annotations(run_dir)
    _mutate_metric(run_dir, "caption_token_multiset_match_rate", 0.0)

    with pytest.raises(ValueError, match="deprecated metric"):
        audit_winoground_run(
            run_dir,
            expected_samples=400,
            annotation_path=annotation_path,
            require_annotation_alignment=True,
            write_outputs=False,
        )


def test_audit_winoground_cli_wrapper_help() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_winoground_run.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert "expected-samples" in result.stdout


def test_audit_winoground_cli_verifies_resolved_annotation(
    tmp_path: Path, research_config
) -> None:
    run_dir = _run_fixture(tmp_path, research_config)
    annotation_path = _write_normalized_annotations(run_dir)
    config_path = run_dir / "config.resolved.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"]["annotation_file"] = str(annotation_path)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_winoground_run.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(run_dir),
            "--expected-samples",
            "3",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout)
    assert summary["annotation_alignment_verified"] is True
    assert summary["all_recomputed_metrics_verified"] is True


def _run_fixture(tmp_path: Path, research_config, *, sample_count: int = 3) -> Path:
    from recoalign.experiments.records import create_run, finalize_run
    from recoalign.reproducibility import atomic_write_json

    config = research_config
    config["data"]["dataset"] = "winoground"
    manifest_path = Path(config["data"]["manifest"])
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["name"] = "winoground"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    run_dir = create_run(config, output_root=tmp_path / "runs", run_id="winoground-test")
    if sample_count == 3:
        scores = np.asarray(
            [
                [[0.9, 0.1], [0.2, 0.8]],
                [[0.4, 0.6], [0.3, 0.7]],
                [[0.5, 0.5], [0.5, 0.5]],
            ],
            dtype=np.float64,
        )
        tags = [["spatial"], ["relation"], ["tie"]]
    else:
        scores = np.tile(
            np.asarray([[[0.9, 0.1], [0.2, 0.8]]], dtype=np.float64),
            (sample_count, 1, 1),
        )
        tags = [["synthetic"] for _ in range(sample_count)]
    result = evaluate_paired_matrix_scores(scores)
    categories = ["winoground"] * sample_count
    metrics = summarize_paired_matrix(result, categories, tags)
    metrics["caption_alphanumeric_character_multiset_match_rate"] = 100.0
    metrics["caption_token_multiset_match_rate"] = 100.0
    finalize_run(run_dir, metrics, status="complete")

    predictions = [
        {
            "sample_id": f"winoground-{index:06d}",
            "category": categories[index],
            "tags": tags[index],
            "scores": [float(value) for value in scores[index].reshape(-1)],
            "image_to_text_correct": bool(result.image_to_text_correct[index]),
            "text_to_image_correct": bool(result.text_to_image_correct[index]),
            "group_correct": bool(result.group_correct[index]),
            "tie": bool(result.ties[index]),
        }
        for index in range(sample_count)
    ]
    _write_jsonl(run_dir / "predictions.jsonl", predictions)
    evaluation = {
        "schema_version": 1,
        "created_at": "2026-07-16T00:00:00+00:00",
        "metrics": metrics,
        "metadata": {
            "benchmark": "winoground_paired_matrix",
            "num_samples": sample_count,
        },
        "predictions_file": "predictions.jsonl",
    }
    atomic_write_json(run_dir / "evaluation.json", evaluation)
    return run_dir


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_normalized_annotations(run_dir: Path) -> Path:
    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    path = run_dir / "normalized-test.jsonl"
    rows = [
        {
            "sample_id": row["sample_id"],
            "category": row["category"],
            "tags": row["tags"],
            "caption_0": "synthetic caption",
            "caption_1": "caption synthetic",
        }
        for row in predictions
    ]
    _write_jsonl(path, rows)
    return path


def _mutate_metric(run_dir: Path, name: str, value: float) -> None:
    metrics_path = run_dir / "metrics.json"
    evaluation_path = run_dir / "evaluation.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics[name] = value
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["metrics"] = metrics
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
