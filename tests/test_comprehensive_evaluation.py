from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from recoalign.evaluation.vlm_benchmark.adapters import load_benchmark_dataset
from recoalign.evaluation.vlm_benchmark.analysis import (
    capability_preservation,
    compare_failure_taxonomy,
    mechanism_consistency,
    structure_token_analysis,
)
from recoalign.evaluation.vlm_benchmark.matrix import (
    build_evaluation_matrix,
    load_ablation_matrix,
    load_matrix_config,
    materialize_matrix_configs,
    validate_evaluation_matrix,
)
from recoalign.evaluation.vlm_benchmark.reporting import generate_comprehensive_reports
from recoalign.evaluation.vlm_benchmark.runner import run_benchmark_cell
from recoalign.models.vlm.base import BaseVLM


class MockBenchmarkVLM(BaseVLM):
    model_id = "mock-benchmark-vlm"

    def load(self):
        self._loaded = True
        return self

    def generate(self, prompt: str, *, image=None, **kwargs) -> str:
        del prompt, image, kwargs
        return "B"

    def generate_with_interface(
        self,
        prompt: str,
        *,
        image,
        interface_checkpoint,
        interface_config=None,
        **kwargs,
    ) -> str:
        del prompt, image, interface_checkpoint, interface_config, kwargs
        return "A"

    @property
    def supports_interface_injection(self) -> bool:
        return True

    def encode_image(self, images) -> np.ndarray:
        return np.ones((len(images), 4), dtype=np.float32) / 2.0


def fixture_config(tmp_path: Path, method: str, seed: int) -> dict:
    image_root = tmp_path / "images"
    image_root.mkdir(exist_ok=True)
    for name in ("a.png", "b.png"):
        Image.new("RGB", (8, 8), (100, 120, 140)).save(image_root / name)
    annotation = tmp_path / "qa.jsonl"
    if not annotation.is_file():
        rows = [
            {
                "sample_id": "a",
                "image": "a.png",
                "question": "Which option is correct?",
                "choices": ["correct", "wrong"],
                "answer": "A",
                "answer_index": 0,
                "category": "relation",
                "dimension": "relation",
            },
            {
                "sample_id": "b",
                "image": "b.png",
                "question": "Which option is correct?",
                "choices": ["correct", "wrong"],
                "answer": "A",
                "answer_index": 0,
                "category": "attribute",
                "dimension": "attribute",
            },
        ]
        annotation.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("schema_version: 1\nname: fixture\nfiles: []\n", encoding="utf-8")
    checkpoint = tmp_path / "interface.pt"
    checkpoint.write_bytes(b"fixture-checkpoint")
    return {
        "schema_version": 1,
        "output_root": str(tmp_path / "outputs"),
        "model": {"name": "reference"},
        "method": {
            "name": method,
            "interface_checkpoint": str(checkpoint) if method == "recoalign" else None,
        },
        "benchmark": {
            "name": "gqa",
            "category": "general_reasoning",
            "format": "qa",
            "manifest": str(manifest),
            "annotation_file": str(annotation),
            "image_root": str(image_root),
            "split": "test",
            "expected_samples": 2,
        },
        "seed": seed,
        "prompt": {"instruction": "Use the shared prompt."},
        "generation": {"temperature": 0.0, "do_sample": False, "max_new_tokens": 4},
        "statistics": {"minimum_seeds": 3},
        "protocol_lock": {"allow_split_override": False},
    }


def test_qa_adapter_and_unified_runner_use_base_vlm(tmp_path) -> None:
    config = fixture_config(tmp_path, "original_vlm", 101)
    dataset = load_benchmark_dataset(config)
    assert len(dataset.samples) == 2
    assert dataset.samples[0].choices == ("A", "B")
    output = run_benchmark_cell(
        config,
        output_dir=tmp_path / "cell",
        model=MockBenchmarkVLM(),
        capture_environment_metadata=False,
    )
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["accuracy"] == 0.0
    assert metrics["compositional"]["relation"]["n"] == 1
    assert (output / "checkpoint_manifest.yaml").is_file()


def test_recoalign_dispatch_requires_hidden_context_and_uses_checkpoint(tmp_path) -> None:
    config = fixture_config(tmp_path, "recoalign", 101)
    output = run_benchmark_cell(
        config,
        output_dir=tmp_path / "recoalign-cell",
        model=MockBenchmarkVLM(),
        capture_environment_metadata=False,
    )
    metrics = json.loads((output / "metrics.json").read_text())
    report = json.loads((output / "report.json").read_text())
    assert metrics["accuracy"] == 1.0
    assert report["scientific_evidence_eligible"] is False
    assert report["oracle_graph_at_inference"] is False


def test_complete_matrix_is_non_selective_and_tracks_blockers() -> None:
    config = load_matrix_config()
    validation = validate_evaluation_matrix(config)
    plan = build_evaluation_matrix(config)
    assert validation["minimum_cells"] == 432
    assert len(plan["cells"]) == 432
    assert plan["no_selective_reporting"]
    assert plan["counts"]["blocked"] > 0


def test_ablation_matrix_and_materialized_cells_are_complete(tmp_path) -> None:
    ablation = load_ablation_matrix()
    assert len(ablation["resolved_matrix"]["methods"]) == 10
    matrix = load_matrix_config()
    manifest = materialize_matrix_configs(matrix, tmp_path / "cells")
    assert manifest["config_count"] == 432
    assert manifest["complete"]
    assert (tmp_path / "cells" / "manifest.yaml").is_file()


def test_analysis_reports_capability_failures_tokens_and_claim_scope() -> None:
    original = [
        {"sample_id": "a", "correct": False, "failure_type": "relation_confusion"},
        {"sample_id": "b", "correct": True, "failure_type": None},
    ]
    method = [
        {"sample_id": "a", "correct": True, "failure_type": None},
        {"sample_id": "b", "correct": True, "failure_type": None},
    ]
    preservation = capability_preservation(original, method)
    failures = compare_failure_taxonomy(original, method)
    assert preservation["preserved"]
    assert failures["reduction"]["relation_confusion"] == 1
    tokens = np.asarray(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.9, 0.1], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ]
    )
    similarity = structure_token_analysis(tokens, ["x", "x", "y"])
    assert similarity["separation"] > 0
    mechanism = mechanism_consistency(
        {
            "claim_status": "infrastructure_validation",
            "scores": {"sas": {"mean": 0.9}, "stas": {"mean": 0.4}},
        },
        [{"gain": 0.1}],
    )
    assert mechanism["diagnosed_interface_gap"] is False
    assert mechanism["scientific_status"] == "pending_real_diagnosis"


def test_reporting_keeps_incomplete_matrix_inconclusive(tmp_path) -> None:
    results = tmp_path / "results"
    model = MockBenchmarkVLM()
    for seed in (101, 202, 303):
        for method in ("original_vlm", "recoalign"):
            config = fixture_config(tmp_path, method, seed)
            run_benchmark_cell(
                config,
                output_dir=results / method / str(seed),
                model=model,
                capture_environment_metadata=False,
            )
    matrix = {
        "schema_version": 1,
        "models": ["reference"],
        "benchmarks": [fixture_config(tmp_path, "original_vlm", 101)["benchmark"]],
        "methods": ["original_vlm", "recoalign"],
        "seeds": [101, 202, 303],
        "statistics": {"minimum_seeds": 3},
        "capability_preservation": {"maximum_degradation": 0.02},
        "protocol_lock": {"allow_split_override": False},
    }
    report = generate_comprehensive_reports(
        matrix_config=matrix,
        results_root=results,
        output_root=tmp_path / "reports",
    )
    assert report["decision"] == "INCONCLUSIVE"
    assert report["complete_cells"] == 6
    decision = json.loads((tmp_path / "reports" / "decision_report.json").read_text())
    assert decision["criteria"]["consistent_paired_gain"]
    assert (tmp_path / "reports" / "latex" / "table_main_results.tex").is_file()
    assert (tmp_path / "reports" / "figures" / "figure_5_failure.svg").is_file()
