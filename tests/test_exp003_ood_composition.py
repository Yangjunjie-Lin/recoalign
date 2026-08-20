from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import yaml

from experiments.ood_composition.runner import run, run_seed_config
from experiments.runtime import load_config
from recoalign.research_registry import get_registered_experiment, validate_research_registries
from recoalign.synthetic_world import GeneratorConfig, SyntheticWorldGenerator
from recoalign.synthetic_world.splits import build_ood_split_suite


def _development_config() -> dict:
    config = deepcopy(load_config("configs/ood_composition.yaml"))
    config["experiment"]["seed"] = 101
    config["experiment"]["seeds"] = [101, 102, 103]
    config["synthetic"]["train_count"] = 8
    config["synthetic"]["test_count"] = 8
    config["synthetic"]["write_images"] = False
    config["evaluation"]["critical"] = False
    config["evaluation"]["bootstrap_samples"] = 100
    return config


def test_controlled_splits_hold_out_only_registered_compositions() -> None:
    generator = SyntheticWorldGenerator(GeneratorConfig(seed=17, write_images=False))
    suite = build_ood_split_suite(generator, train_count=8, test_count=8, seed=17)
    assert suite.validation["valid"] is True
    assert suite.validation["assertions"] == {
        "composition_overlap": False,
        "relation_combination_overlap": False,
        "hop_depth_overlap": False,
        "primitive_coverage": True,
        "sample_id_overlap": False,
    }
    assert suite.splits["composition_ood"].validation["composition_overlap"] == []
    assert "left+behind" in suite.splits["relation_ood"].validation[
        "test_relation_combinations"
    ]
    assert suite.splits["relation_ood"].validation["relation_combination_overlap"] == []
    assert suite.splits["hop_ood"].validation["train_hop_depths"] == [1, 2]
    assert suite.splits["hop_ood"].validation["test_hop_depths"] == [3, 4]


def test_split_generation_is_reproducible() -> None:
    generator = SyntheticWorldGenerator(GeneratorConfig(seed=23, write_images=False))
    first = build_ood_split_suite(generator, train_count=8, test_count=8, seed=23)
    second = build_ood_split_suite(generator, train_count=8, test_count=8, seed=23)
    for name in first.splits:
        for partition in ("train", "test"):
            left = getattr(first.splits[name], partition)
            right = getattr(second.splits[name], partition)
            assert [row.to_dict() for row in left] == [row.to_dict() for row in right]


def test_materialized_split_manifest_hashes_every_declared_file(tmp_path: Path) -> None:
    root = tmp_path / "splits"
    generator = SyntheticWorldGenerator(GeneratorConfig(seed=29, write_images=False))
    suite = build_ood_split_suite(
        generator, train_count=8, test_count=8, seed=29, output_dir=root
    )
    assert suite.manifest is not None
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["validation"]["valid"] is True
    for split in manifest["splits"].values():
        split_manifest = json.loads((root / split["path"]).read_text(encoding="utf-8"))
        assert split_manifest["files"]
        for entry in split_manifest["files"]:
            path = root / entry["path"]
            assert path.stat().st_size == entry["bytes"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


def test_seed_runner_emits_all_ood_metrics_and_integrity_controls(tmp_path: Path) -> None:
    metrics = run_seed_config(_development_config(), output_dir=tmp_path / "seed", seed=101)
    assert metrics["integrity"]["valid"] is True
    assert metrics["integrity"]["random_structure_control"] == {
        "random_graph_not_equivalent": True,
        "random_graph_length_matched": True,
        "n_pairs": 64,
        "maximum_token_delta": 0,
    }
    assert set(metrics["splits"]) == {"iid", "composition_ood", "relation_ood", "hop_ood"}
    assert set(metrics["reasoning_depth"]["depths"]) == {"1", "2", "3", "4"}
    assert set(metrics["anti_memorization"]) == {
        "object_identity_swap",
        "relation_recombination",
        "attribute_transfer",
    }
    rows = [
        json.loads(line)
        for line in (tmp_path / "seed" / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["condition"] for row in rows} == {
        "image_only",
        "caption",
        "scene_graph",
        "random_graph",
    }
    assert all(row["pair_id"] and row["evaluation_role"] for row in rows)


def test_exp003_multiseed_bundle_and_registry_integration(tmp_path: Path) -> None:
    config = _development_config()
    config_path = tmp_path / "exp003.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    output = tmp_path / "EXP003"
    result = run(config_path, output_dir=output)
    assert result["experiment_id"] == "EXP003"
    assert result["decision"] == "INCONCLUSIVE"
    required = {
        "config.resolved.yaml",
        "metrics.json",
        "predictions.jsonl",
        "split_manifest.json",
        "run.json",
        "decision_report.yaml",
        "figure1_iid_vs_ood_performance.png",
        "figure2_generalization_gap.png",
        "figure3_reasoning_depth.png",
        "implementation_report.md",
        "ood_benchmark_report.md",
        "generalization_analysis.md",
    }
    assert required <= {path.name for path in output.iterdir()}
    decision = yaml.safe_load((output / "decision_report.yaml").read_text(encoding="utf-8"))
    assert decision["evidence"]["evidence_role"] == "infrastructure_validation"
    assert decision["robustness_check"]["model_scientific_decision_allowed"] is False
    assert validate_research_registries()["valid"] is True
    registration = get_registered_experiment("EXP003")
    assert registration["dataset"]["version"] == "generator-v2"
    assert registration["input_conditions"][-1] == "random_graph"
    assert len(registration["decision_rule"]["criteria"]) == 4
