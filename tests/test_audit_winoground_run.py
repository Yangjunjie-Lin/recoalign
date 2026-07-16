from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import yaml

from recoalign.evaluation.diagnostics import (
    evaluate_paired_matrix_scores,
    summarize_paired_matrix,
)

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_winoground_run.py"


def _load_auditor() -> ModuleType:
    spec = importlib.util.spec_from_file_location("audit_winoground_run", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDITOR = _load_auditor()


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


def _run_fixture(tmp_path: Path, research_config) -> Path:
    from recoalign.experiments.records import create_run, finalize_run
    from recoalign.reproducibility import atomic_write_json

    config = research_config
    config["data"]["dataset"] = "winoground"
    manifest_path = Path(config["data"]["manifest"])
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["name"] = "winoground"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    run_dir = create_run(config, output_root=tmp_path / "runs", run_id="winoground-test")
    scores = np.asarray(
        [
            [[0.9, 0.1], [0.2, 0.8]],
            [[0.4, 0.6], [0.3, 0.7]],
            [[0.5, 0.5], [0.5, 0.5]],
        ],
        dtype=np.float64,
    )
    result = evaluate_paired_matrix_scores(scores)
    categories = ["winoground"] * 3
    tags = [["spatial"], ["relation"], ["tie"]]
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
        for index in range(3)
    ]
    _write_jsonl(run_dir / "predictions.jsonl", predictions)
    evaluation = {
        "schema_version": 1,
        "created_at": "2026-07-16T00:00:00+00:00",
        "metrics": metrics,
        "metadata": {
            "benchmark": "winoground_paired_matrix",
            "num_samples": 3,
        },
        "predictions_file": "predictions.jsonl",
    }
    atomic_write_json(run_dir / "evaluation.json", evaluation)
    return run_dir


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
