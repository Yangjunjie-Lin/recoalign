from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from experiments.graph_vs_text.runner import run
from experiments.runtime import load_config
from models.vlm.base import ReferenceVLM
from recoalign.synthetic_world.questions.conditions import (
    ABLATION_CONDITIONS,
    controlled_evidence_view,
    evidence_view,
)
from synthetic_world.generator.generator import GeneratorConfig, SyntheticWorldGenerator


def _record():
    return SyntheticWorldGenerator(GeneratorConfig(seed=71, write_images=False)).generate(1)[0]


def test_caption_and_graph_are_semantically_and_token_controlled() -> None:
    record = _record()
    caption = evidence_view(record, "caption")
    graph = evidence_view(record, "scene_graph")
    assert caption.semantic_facts_sha256 == graph.semantic_facts_sha256
    assert caption.semantic_units == graph.semantic_units
    assert caption.relation_count == graph.relation_count

    matched_caption = controlled_evidence_view(record, "caption", setting="token_matched")
    matched_graph = controlled_evidence_view(record, "scene_graph", setting="token_matched")
    assert abs(matched_caption.matched_tokens - matched_graph.matched_tokens) <= 1
    assert matched_caption.semantic_units == matched_graph.semantic_units


def test_required_ablations_preserve_declared_full_fact_hash() -> None:
    record = _record()
    expected = evidence_view(record, "scene_graph").semantic_facts_sha256
    assert set(ABLATION_CONDITIONS) == {
        "graph_order_shuffled",
        "text_unordered",
        "graph_serialization_triples",
        "graph_serialization_json",
    }
    for condition in ABLATION_CONDITIONS:
        assert evidence_view(record, condition).semantic_facts_sha256 == expected
    assert (
        evidence_view(record, "graph_order_shuffled").text
        != evidence_view(record, "scene_graph").text
    )


def test_base_vlm_prepare_generate_evaluate_interface() -> None:
    record = _record()
    model = ReferenceVLM(seed=71, condition_accuracy={"caption": 1.0})
    prepared = model.prepare_input(record, "caption", setting="token_matched")
    prediction = model.reason(record, "caption", prepared_input=prepared)
    assert prepared.input_tokens >= prepared.evidence_tokens
    assert prepared.semantic_facts_sha256
    assert model.evaluate(prediction, record) is True
    assert isinstance(model.generate(prepared.prompt, image=prepared.image), str)


def test_llava_stage_two_config_is_explicit_local_autoload_template() -> None:
    config = load_config("configs/exp001_llava15.yaml")
    assert config["model"]["backend"] == "llava"
    assert config["model"]["auto_load"] is True
    assert config["model"]["model_dir"] == "outputs/models/llava-v1.5-7b"


def test_multiseed_exp001_writes_complete_inconclusive_reference_bundle(
    tmp_path: Path,
) -> None:
    config = load_config("configs/graph_vs_text.yaml")
    config["experiment"]["seed"] = 101
    config["experiment"]["seeds"] = [101, 102, 103]
    config["synthetic"]["count"] = 12
    config["synthetic"]["write_images"] = False
    config["ood"]["train_count"] = 8
    config["ood"]["test_count"] = 8
    config["evaluation"]["critical"] = False
    config["evaluation"]["bootstrap_samples"] = 100
    config_path = tmp_path / "exp001.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "EXP001"

    result = run(config_path, output_dir=output)

    assert result["decision"] == "INCONCLUSIVE"
    required = {
        "config.resolved.yaml",
        "metrics.json",
        "predictions.jsonl",
        "run.json",
        "manifest.json",
        "decision_report.yaml",
        "figure1_performance_comparison.png",
        "figure2_reasoning_depth.png",
        "figure3_token_efficiency.png",
    }
    assert required <= {path.name for path in output.iterdir()}
    stored = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert len(stored["metrics"]["per_seed"]) == 3
    assert stored["decision"] == "INCONCLUSIVE"
    decision = yaml.safe_load((output / "decision_report.yaml").read_text(encoding="utf-8"))
    assert decision["evidence"]["evidence_role"] == "infrastructure_validation"
    assert decision["robustness_check"]["model_scientific_decision_allowed"] is False
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"]["predictions.jsonl"]["sha256"]


def test_failed_target_backend_run_is_retained_as_inconclusive(tmp_path: Path) -> None:
    config = load_config("configs/graph_vs_text.yaml")
    config["experiment"]["seeds"] = [201, 202, 203]
    config["synthetic"]["count"] = 1
    config["synthetic"]["write_images"] = False
    config["evaluation"]["critical"] = False
    config["model"]["backend"] = "llava"
    config["model"]["model_dir"] = str(tmp_path / "missing-checkpoint")
    config["model"].pop("condition_accuracy")
    config_path = tmp_path / "failed.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "failed-EXP001"

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        run(config_path, output_dir=output)

    run_record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run_record["status"] == "failed"
    assert run_record["decision"] == "INCONCLUSIVE"
    decision = yaml.safe_load((output / "decision_report.yaml").read_text(encoding="utf-8"))
    assert decision["final_decision"] == "INCONCLUSIVE"
