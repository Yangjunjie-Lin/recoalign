from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from recoalign.synthetic_world.evaluation import evaluate_synthetic, seed_summary


def test_seed_summary_reports_mean_std_and_ci() -> None:
    summary = seed_summary([0.6, 0.7, 0.8])
    assert summary["mean"] == pytest.approx(0.7)
    assert summary["std"] > 0
    assert summary["ci95"][0] <= summary["mean"] <= summary["ci95"][1]
    assert summary["n_seeds"] == 3


def test_evaluate_synthetic_writes_required_artifacts(tmp_path: Path) -> None:
    config = {
        "benchmark": {
            "name": "synthetic-test",
            "seed": 41,
            "count": 8,
            "output_dir": str(tmp_path / "ignored"),
            "split_strategy": "composition",
            "resolution": 128,
            "style": "flat",
            "write_images": False,
            "min_hops": 1,
            "max_hops": 4,
        },
        "model": {
            "backend": "reference",
            "condition_accuracy": {"caption": 0.7, "scene_graph": 0.85},
        },
        "evaluation": {
            "critical": False,
            "seeds": [41, 42, 43],
            "confidence_level": 0.95,
            "conditions": ["caption", "scene_graph"],
        },
        "training": {"enabled": False},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "run"
    metrics = evaluate_synthetic(config_path, output_dir=output)
    assert metrics["seed_count"] == 3
    assert {"metrics.json", "predictions.jsonl", "decision_report.yaml"} <= {
        path.name for path in output.iterdir()
    }
    stored = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert {"mean", "std", "ci95"} <= stored["conditions"]["caption"].keys()
    decision = yaml.safe_load((output / "decision_report.yaml").read_text(encoding="utf-8"))
    assert decision["instrument_validation"] == "PASS"
    assert decision["scientific_decision"] == "NOT_APPLICABLE"
