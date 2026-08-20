from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from jsonschema import ValidationError

from experiments.runtime import build_vlm, load_config, run_conditions
from recoalign.cli import build_parser
from recoalign.models.vlm.base import BaseVLM
from recoalign.models.vlm.evaluation import AnswerEvaluator
from recoalign.models.vlm.failure_analysis import write_failure_analysis
from recoalign.models.vlm.llava import Llava15VLM
from recoalign.models.vlm.prompting import load_prompt_protocol
from recoalign.models.vlm.registry import (
    ModelRegistry,
    dry_run_model,
    validate_model_definition,
)
from recoalign.synthetic_world import GeneratorConfig, SyntheticWorldGenerator
from recoalign.vlm_evaluation import compatibility_matrix, run_vlm_evaluation


class MockBackend:
    def __init__(self) -> None:
        self.loads = 0

    def load(self) -> None:
        self.loads += 1

    def generate(self, prompt: str, *, image=None, **kwargs) -> str:
        del prompt, image, kwargs
        return "left"

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def encode_image(self, images) -> list[list[float]]:
        return [[1.0, 0.0] for _ in images]


def test_cli_import_does_not_require_torch() -> None:
    script = """
import builtins

original_import = builtins.__import__

def blocked_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("blocked for optional-dependency test")
    return original_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
from recoalign.cli import build_parser
build_parser()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_synthetic_benchmark_config_validation_does_not_require_torch() -> None:
    script = """
import builtins
import sys

original_import = builtins.__import__

def blocked_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("blocked for optional-dependency test")
    return original_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
from recoalign.cli import main
raise SystemExit(main(["validate-config", "configs/synthetic_benchmark.yaml"]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_base_vlm_exposes_complete_unified_lifecycle() -> None:
    for method in ("load", "encode_image", "prepare_input", "generate", "evaluate"):
        assert hasattr(BaseVLM, method)


def test_mock_llava_inference_is_lazy_and_loads_once(tmp_path: Path) -> None:
    backend = MockBackend()
    adapter = Llava15VLM(model_dir=tmp_path, backend=backend)
    assert adapter.loaded is False
    assert adapter.generate("prompt") == "left"
    assert adapter.generate("prompt") == "left"
    assert adapter.loaded is True
    assert backend.loads == 1
    assert adapter.encode_image(["a", "b"]).shape == (2, 2)


def test_shared_prompt_is_byte_equivalent_to_phase1_protocol() -> None:
    prompt = load_prompt_protocol("configs/prompts/reasoning_default.yaml")
    assert prompt.render("Evidence: image only.", "Where?", ("left", "right", "above")) == (
        "Evidence: image only.\nQuestion: Where?\n"
        "Answer with exactly one option: left, right, above."
    )


def test_uniform_answer_evaluator_modes() -> None:
    evaluator = AnswerEvaluator()
    assert evaluator.evaluate("left", "left", choices=("left", "right", "above")).method == "exact"
    assert evaluator.evaluate(" Left. ", "left", choices=("left", "right", "above")).correct
    traced = evaluator.evaluate(
        "I inspect the graph. Final answer: behind",
        "behind",
        choices=("front", "behind", "left"),
    )
    assert traced.correct is True
    assert traced.reasoning_trace is not None
    ambiguous = evaluator.evaluate(
        "left or right", "left", choices=("left", "right", "above")
    )
    assert ambiguous.correct is False


def test_registry_covers_full_model_experiment_matrix() -> None:
    matrix = compatibility_matrix()
    assert set(matrix["models"]) == {
        "reference",
        "llava_1_5_7b",
        "llava_next",
        "qwen_vl",
        "internvl",
    }
    for model in matrix["models"].values():
        assert set(model["experiments"]) == {"EXP001", "EXP002", "EXP003"}
        assert all(cell["compatible"] for cell in model["experiments"].values())
    assert matrix["models"]["llava_1_5_7b"]["scientific_evidence"] is True
    assert matrix["models"]["qwen_vl"]["scientific_evidence"] is False


def test_model_pool_reuses_real_weights_but_not_seeded_reference() -> None:
    registry = ModelRegistry()
    llava = registry.definition("llava_1_5_7b").experiment_model_config()
    assert registry.get_or_create(llava, seed=1) is registry.get_or_create(llava, seed=2)
    reference = registry.definition("reference").experiment_model_config()
    assert registry.get_or_create(reference, seed=1) is not registry.get_or_create(
        reference, seed=2
    )


def test_reference_evaluation_emits_uniform_prediction_fields() -> None:
    config = load_config("configs/graph_vs_text.yaml")
    model = build_vlm(config)
    records = SyntheticWorldGenerator(GeneratorConfig(write_images=False)).generate(2)
    rows = run_conditions(records, model, ("caption", "scene_graph"))
    assert len(rows) == 4
    assert all(row["sample_id"] == row["scene_id"] for row in rows)
    assert all(row["ground_truth"] == row["answer"] for row in rows)
    assert all(row["evaluation"]["evaluation_method"] for row in rows)


def test_llava_dry_run_validates_manifest_without_loading_weights(tmp_path: Path) -> None:
    registry = ModelRegistry()
    config = registry.definition("llava_1_5_7b").experiment_model_config()
    config["model_path"] = str(tmp_path / "llava-v1.5-7b")
    config["tokenizer_path"] = config["model_path"]
    report = dry_run_model(config)
    assert report["weights_loaded"] is False
    assert report["checkpoint_exists"] is False
    assert report["checkpoint_files"] == []
    assert report["checkpoint_files_declared"]
    assert report["checkpoint_manifest_loaded"] is True
    assert report["checkpoint_manifest_exists"] is True
    assert report["checkpoint_format"] == "legacy_llava"
    assert report["checkpoint_format_source"] == "checkpoint_manifest"
    assert report["loader"] == "legacy_transformers"
    assert report["loader_compatible"] is True
    assert report["processor_ready"] is False
    assert report["processor_contract_ready"] is True
    assert report["runtime_ready"] is False


def test_llava_runtime_rejects_missing_checkpoint_after_manifest_dry_run(tmp_path: Path) -> None:
    registry = ModelRegistry()
    config = registry.definition("llava_1_5_7b").experiment_model_config()
    config["model_path"] = str(tmp_path / "llava-v1.5-7b")
    config["tokenizer_path"] = config["model_path"]
    model = Llava15VLM.from_config(config)
    with pytest.raises(FileNotFoundError, match="download the pinned checkpoint"):
        model.load()


def test_vlm_eval_dry_run_writes_required_bundle(tmp_path: Path) -> None:
    output = run_vlm_evaluation(
        model_name="llava_1_5_7b",
        experiment_id="EXP001",
        split="test",
        seeds=[101],
        output_dir=tmp_path / "dry-run",
        dry_run=True,
    )
    assert {
        "config.resolved.yaml",
        "metrics.json",
        "predictions.jsonl",
        "run.json",
        "manifest.json",
    } <= {path.name for path in output.iterdir()}
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "dry-run"
    assert run["checkpoint"]["checkpoint_fingerprint"]
    assert run["checkpoint"]["revision"] == "4481d270cc22fd5c4d1bb5df129622006ccd9234"
    assert run["runtime"]["torch"]
    validation = run["dry_run_validation"]
    assert validation["weights_loaded"] is False
    assert validation["checkpoint_manifest_loaded"] is True
    expected_runtime_ready = (
        validation["checkpoint_exists"]
        and bool(validation["checkpoint_files"])
        and validation["checkpoint_manifest_error"] is None
        and all(validation["dependencies"].values())
        and validation["hardware_ready"]
        and validation["processor_ready"]
        and validation["loader_compatible"]
    )
    assert validation["runtime_ready"] is expected_runtime_ready


def test_vlm_eval_rejects_posthoc_split_changes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="registered benchmark split"):
        run_vlm_evaluation(
            model_name="llava_1_5_7b",
            experiment_id="EXP001",
            split="ood_test",
            output_dir=tmp_path / "invalid",
            dry_run=True,
        )


def test_model_config_validation_fails_closed() -> None:
    payload = yaml.safe_load(Path("configs/models/llava_1_5_7b.yaml").read_text())
    payload["generation"]["temperature"] = 0.7
    with pytest.raises((ValidationError, ValueError)):
        validate_model_definition(payload, expected_name="llava_1_5_7b")


def test_failure_analysis_assigns_registered_taxonomy(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        {
            "sample_id": "a",
            "scene_id": "a",
            "condition": "caption",
            "question_type": "relation_reasoning",
            "hop_depth": 1,
            "prediction": "left",
            "ground_truth": "left",
            "correct": True,
        },
        {
            "sample_id": "a",
            "scene_id": "a",
            "condition": "scene_graph",
            "question_type": "relation_reasoning",
            "hop_depth": 1,
            "prediction": "right",
            "ground_truth": "left",
            "correct": False,
        },
    ]
    predictions.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    summary = write_failure_analysis(predictions, tmp_path / "failures")
    assert summary["failure_types"] == {"structure_misuse": 1}
    assert (tmp_path / "failures" / "errors.jsonl").is_file()


def test_cli_supports_both_vlm_entrypoint_forms() -> None:
    parser = build_parser()
    direct = parser.parse_args(
        ["run-vlm-eval", "--model", "llava_1_5_7b", "--experiment", "EXP001"]
    )
    governed = parser.parse_args(
        [
            "run-experiment",
            "--experiment",
            "EXP001",
            "--model",
            "llava_1_5_7b",
            "--dry-run",
        ]
    )
    assert direct.experiment == governed.experiment_option == "EXP001"
