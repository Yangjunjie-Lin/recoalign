from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import yaml

from evaluation.statistics import paired_bootstrap_test
from experiments.graph_ablation.runner import run, run_seed_config
from experiments.runtime import load_config
from recoalign.research_registry import get_registered_experiment, validate_research_registries
from recoalign.synthetic_world.corruption import (
    flip_relation,
    randomize_graph,
    remove_relation,
    swap_entity,
)
from recoalign.synthetic_world.scene_graph import SceneGraph


def _graph() -> SceneGraph:
    nodes = tuple(
        {
            "id": f"obj{index}",
            "category": "object",
            "shape": shape,
            "color": color,
            "size": "medium",
            "texture": "solid",
        }
        for index, (shape, color) in enumerate(
            zip(
                ("circle", "square", "triangle", "cube", "sphere"),
                ("red", "blue", "green", "yellow", "purple"),
                strict=True,
            ),
            start=1,
        )
    )
    edges = tuple(
        {
            "subject": f"obj{index}",
            "relation": "left",
            "object": f"obj{index + 1}",
        }
        for index in range(1, 5)
    )
    return SceneGraph(nodes, edges)


def _development_config() -> dict:
    config = deepcopy(load_config("configs/graph_ablation.yaml"))
    config["experiment"]["seeds"] = [101, 102, 103]
    config["experiment"]["seed"] = 101
    config["synthetic"]["count"] = 16
    config["synthetic"]["write_images"] = False
    config["evaluation"]["critical"] = False
    config["evaluation"]["bootstrap_samples"] = 100
    return config


def test_relation_removal_ratios_are_controlled_and_support_aware() -> None:
    graph = _graph()
    expected_removed = {0.25: 1, 0.50: 2, 0.75: 3}
    for ratio, count in expected_removed.items():
        result = remove_relation(
            graph,
            ratio=ratio,
            seed=17,
            critical_edge_indices=(0, 1, 2, 3),
        )
        assert len(graph.edges) - len(result.corrupted.edges) == count
        assert len(result.selected_edge_indices) == count
        assert result.to_manifest()["preserves_nodes"] is True
        assert result.to_manifest()["changed"] is True


def test_wrong_graph_operators_preserve_declared_invariants() -> None:
    graph = _graph()
    flipped = flip_relation(graph, seed=9)
    swapped = swap_entity(graph, seed=9)
    randomized = randomize_graph(graph, seed=9)
    for result in (flipped, swapped, randomized):
        manifest = result.to_manifest()
        assert manifest["changed"] is True
        assert manifest["preserves_nodes"] is True
        assert manifest["preserves_edge_count"] is True
        assert len(result.corrupted.edges) == len(graph.edges)
    assert {edge["relation"] for edge in flipped.corrupted.edges} == {"right"}
    assert all(edge["subject"] != edge["object"] for edge in swapped.corrupted.edges)
    assert randomized == randomize_graph(graph, seed=9)
    assert randomized != randomize_graph(graph, seed=10)


def test_paired_bootstrap_is_deterministic_and_directional() -> None:
    first = [1.0, 1.0, 0.8, 0.9, 1.0]
    second = [0.2, 0.3, 0.1, 0.2, 0.4]
    result = paired_bootstrap_test(first, second, samples=200, seed=11)
    assert result == paired_bootstrap_test(first, second, samples=200, seed=11)
    assert result["method"] == "paired_bootstrap"
    assert result["p_value"] < 0.05


def test_seed_runner_has_no_control_metadata_leakage_and_exact_length_match(
    tmp_path: Path,
) -> None:
    config = _development_config()
    output = tmp_path / "seed"
    metrics = run_seed_config(config, output_dir=output, seed=101)
    assert metrics["integrity"]["corruption_manifest_complete"] is True
    assert metrics["integrity"]["original_graph_not_in_prompt_metadata"] is True
    assert metrics["anti_shortcut"]["graph_length_matching"]["passed"] is True

    rows = [
        json.loads(line)
        for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_scene: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_scene.setdefault(row["scene_id"], {})[row["condition"]] = row
        prompt = row["input"]["prompt"]
        assert '"original_graph"' not in prompt
        assert '"corrupted_graph"' not in prompt
        assert '"operation"' not in prompt
    assert all(
        values["scene_graph"]["input"]["input_tokens"]
        == values["random_graph"]["input"]["input_tokens"]
        for values in by_scene.values()
    )
    manifest = json.loads(
        (output / "corruption_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["complete"] is True
    assert manifest["entry_count"] == 16 * 8
    assert all(entry["preserves_nodes"] for entry in manifest["entries"])


def test_seed_runner_is_reproducible(tmp_path: Path) -> None:
    config = _development_config()
    first = tmp_path / "first"
    second = tmp_path / "second"
    run_seed_config(config, output_dir=first, seed=103)
    run_seed_config(config, output_dir=second, seed=103)
    assert (first / "predictions.jsonl").read_text(encoding="utf-8") == (
        second / "predictions.jsonl"
    ).read_text(encoding="utf-8")
    assert (first / "corruption_manifest.json").read_text(encoding="utf-8") == (
        second / "corruption_manifest.json"
    ).read_text(encoding="utf-8")


def test_exp002_bundle_and_registry_integration(tmp_path: Path) -> None:
    config = _development_config()
    config_path = tmp_path / "exp002.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    output = tmp_path / "bundle"
    result = run(config_path, output_dir=output)
    assert result["experiment_id"] == "EXP002"
    assert result["decision"] == "INCONCLUSIVE"
    for name in (
        "config.resolved.yaml",
        "metrics.json",
        "predictions.jsonl",
        "corruption_manifest.json",
        "run.json",
        "decision_report.yaml",
        "figure1_accuracy_degradation.png",
        "figure2_corruption_ratio.png",
        "figure3_reasoning_depth_degradation.png",
        "relation_breakdown.csv",
    ):
        assert (output / name).is_file(), name

    assert validate_research_registries()["valid"] is True
    registration = get_registered_experiment("EXP002")
    assert registration["dataset"]["version"] == "generator-v2"
    assert registration["decision_rule"]["robustness"]["minimum_seeds"] == 5
    assert len(registration["decision_rule"]["criteria"]) == 7
