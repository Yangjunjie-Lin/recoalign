from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from recoalign.experiments.records import create_run, finalize_run
from recoalign.reproducibility import atomic_write_json

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_runs.py"


def _load_comparator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("compare_runs", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COMPARATOR = _load_comparator()


def test_compare_runs_accepts_tiny_score_differences(tmp_path: Path, research_config) -> None:
    cached = _run_fixture(tmp_path, research_config, "cached", score_delta=0.0)
    no_cache = _run_fixture(tmp_path, research_config, "no-cache", score_delta=1e-7)

    summary = COMPARATOR.compare_runs(cached, no_cache, atol=1e-6, rtol=1e-6)

    assert summary["passed"] is True
    assert summary["maximum_score_absolute_difference"] < 1e-6
    assert not any(summary["decision_differences"].values())
    assert summary["no_cache_run_cache_disabled"] is True


def test_compare_runs_rejects_cold_cache_as_no_cache(tmp_path: Path, research_config) -> None:
    cached = _run_fixture(tmp_path, research_config, "cached", score_delta=0.0)
    no_cache = _run_fixture(tmp_path, research_config, "no-cache", score_delta=0.0)
    path = no_cache / "evaluation.json"
    evaluation = json.loads(path.read_text())
    evaluation["metadata"]["cache"]["enabled"] = True
    atomic_write_json(path, evaluation)

    summary = COMPARATOR.compare_runs(cached, no_cache)

    assert summary["passed"] is False
    assert summary["no_cache_run_cache_disabled"] is False


def test_compare_runs_rejects_decision_difference(tmp_path: Path, research_config) -> None:
    cached = _run_fixture(tmp_path, research_config, "cached", score_delta=0.0)
    no_cache = _run_fixture(tmp_path, research_config, "no-cache", score_delta=0.0)
    path = no_cache / "predictions.jsonl"
    prediction = json.loads(path.read_text())
    prediction["tie"] = True
    path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")

    summary = COMPARATOR.compare_runs(cached, no_cache)

    assert summary["passed"] is False
    assert summary["decision_differences"]["tie"] == 1


def _run_fixture(
    tmp_path: Path,
    research_config,
    run_id: str,
    *,
    score_delta: float,
) -> Path:
    config = research_config
    config["data"]["dataset"] = "winoground"
    run_dir = create_run(config, output_root=tmp_path / "runs", run_id=run_id)
    metrics = {
        "image_to_text_accuracy": 100.0,
        "text_to_image_accuracy": 100.0,
        "group_accuracy": 100.0,
        "tie_rate": 0.0,
        "mean_image_to_text_margin": 0.7,
        "mean_text_to_image_margin": 0.6,
        "caption_token_multiset_match_rate": 100.0,
    }
    finalize_run(run_dir, metrics, status="complete")
    prediction = {
        "sample_id": "winoground-000000",
        "category": "winoground",
        "tags": ["spatial"],
        "scores": [0.9 + score_delta, 0.1, 0.2, 0.8],
        "image_to_text_correct": True,
        "text_to_image_correct": True,
        "group_correct": True,
        "tie": False,
    }
    (run_dir / "predictions.jsonl").write_text(
        json.dumps(prediction) + "\n", encoding="utf-8"
    )
    run = json.loads((run_dir / "run.json").read_text())
    metadata = {
        "protocol_version": 1,
        "protocol": "winoground-official-400-v1",
        "dataset": "winoground",
        "split": "test",
        "annotation_sha256": "a" * 64,
        "dataset_manifest_sha256": run["dataset_manifest_sha256"],
        "encoder": {
            "framework": "open_clip",
            "model": "ViT-B-32",
            "pretrained": "laion2b_s34b_b79k",
        },
        "cache": {
            "enabled": run_id != "no-cache",
            "images_hit": False,
            "texts_hit": False,
        },
    }
    evaluation = {
        "schema_version": 1,
        "created_at": "2026-07-16T00:00:00+00:00",
        "metrics": metrics,
        "metadata": metadata,
        "predictions_file": "predictions.jsonl",
    }
    atomic_write_json(run_dir / "evaluation.json", evaluation)
    return run_dir
