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

    summary = audit_winoground_run(run_dir, expected_samples=400, write_outputs=False)

    assert summary["prediction_count"] == 400
    assert summary["metrics_recomputed"] is True
    assert summary["decisions_recomputed"] is True


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
