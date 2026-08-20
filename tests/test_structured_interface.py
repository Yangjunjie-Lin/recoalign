from __future__ import annotations

import json

import pytest

from diagnosis.interface_gap_analysis.gap import summarize_interface_gap
from evaluation.metrics import evaluate_rows
from models.vlm.base import ReferenceVLM
from models.vlm.llava import Llava15VLM
from synthetic_world.compositional_tasks.tasks import condition_text, graph_variant
from synthetic_world.generator.generator import GeneratorConfig, SyntheticWorldGenerator


def test_generator_emits_json_compatible_scene_graph(tmp_path) -> None:
    generator = SyntheticWorldGenerator(GeneratorConfig(seed=11, write_images=False))
    records = generator.generate(4, output_dir=tmp_path / "world")
    assert len(records) == 4
    payload = records[0].to_dict()
    assert payload["graph"]["edges"]
    assert {"shape", "color", "category"} <= payload["objects"][0].keys()
    first_row = json.loads((tmp_path / "world" / "metadata.jsonl").read_text().splitlines()[0])
    assert first_row["scene_id"] == records[0].scene_id


def test_graph_variants_are_explicit() -> None:
    record = SyntheticWorldGenerator(GeneratorConfig(write_images=False)).generate(1)[0]
    assert len(graph_variant(record, "scene_graph").edges) == 2
    assert len(graph_variant(record, "partial_graph").edges) == 1
    assert graph_variant(record, "corrupted_graph").edges != record.relations
    assert "Nodes:" in condition_text(record, "scene_graph")


def test_reference_backend_is_deterministic_and_condition_aware() -> None:
    record = SyntheticWorldGenerator(GeneratorConfig(seed=3, write_images=False)).generate(1)[0]
    backend = ReferenceVLM(seed=3)
    assert backend.reason(record, "scene_graph") == backend.reason(record, "scene_graph")
    assert backend.encode_text(["a", "b"]).shape == (2, 16)


def test_metrics_and_interface_gap_contrasts() -> None:
    rows = []
    for scene_id in ("a", "b"):
        for condition, correct in (("image_only", scene_id == "a"), ("scene_graph", True)):
            rows.append({"scene_id": scene_id, "condition": condition, "correct": correct})
    metrics = evaluate_rows(rows, bootstrap_samples=20)
    assert metrics["conditions"]["scene_graph"]["accuracy"] == 1.0
    gap = summarize_interface_gap(rows)
    assert gap["contrasts"]["structured_reasoning_gain"]["estimate"] == 0.5


def test_ood_compositions_are_disjoint() -> None:
    generator = SyntheticWorldGenerator(GeneratorConfig(seed=5, write_images=False))
    train, test = generator.generate_ood_splits(8, 8, seed=5)
    train_compositions = {row.metadata["composition"] for row in train}
    test_compositions = {row.metadata["composition"] for row in test}
    assert train_compositions.isdisjoint(test_compositions)


def test_ood_generator_supports_registered_split_size() -> None:
    generator = SyntheticWorldGenerator(GeneratorConfig(seed=20260818, write_images=False))
    train, test = generator.generate_ood_splits(60, 60, seed=20260818)
    assert len(train) == 60
    assert len(test) == 60
    assert len({row.metadata["composition"] for row in test}) == 60


def test_ood_generator_rejects_impossible_split_size() -> None:
    generator = SyntheticWorldGenerator(GeneratorConfig(seed=5, write_images=False))
    with pytest.raises(ValueError, match="OOD compositions"):
        generator.generate_ood_splits(1, 360, seed=5)


def test_ood_split_materializes_images_and_manifest(tmp_path) -> None:
    generator = SyntheticWorldGenerator(GeneratorConfig(seed=5, write_images=True))
    train, test = generator.generate_ood_splits(3, 3, output_dir=tmp_path / "ood", seed=5)
    assert all(
        record.image and (tmp_path / "ood" / "train" / "images").exists()
        for record in train
    )
    assert all(
        record.image and (tmp_path / "ood" / "test" / "images").exists() for record in test
    )
    manifest = json.loads((tmp_path / "ood" / "manifest.json").read_text())
    assert manifest["composition_overlap"] is False


def test_checkpoint_provenance_is_recorded(tmp_path) -> None:
    from experiments.runtime import load_config, run_standard_suite

    config = load_config("configs/graph_vs_text.yaml")
    config["experiment"]["output_dir"] = str(tmp_path / "run")
    run_standard_suite(config, name="graph_vs_text", conditions=("scene_graph",), count=2)
    manifest = json.loads((tmp_path / "run" / "run.json").read_text())
    assert manifest["checkpoint"]["status"] == "declared"
    assert len(manifest["checkpoint"]["manifest_sha256"]) == 64
    assert (tmp_path / "run" / "config.resolved.yaml").exists()
    assert (tmp_path / "run" / "run.log").exists()


def test_llava_adapter_does_not_silently_fallback(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    adapter = Llava15VLM.from_pretrained(model_dir)
    with pytest.raises(RuntimeError, match="backend"):
        adapter.generate("hello")
