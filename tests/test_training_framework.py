from __future__ import annotations

import json
from copy import deepcopy

import pytest
import torch
from torch import nn

from recoalign.models.recoalign import ReCoAlignConfig
from recoalign.training.ablations import (
    apply_ablation,
    load_ablation_config,
    randomize_structural_labels,
)
from recoalign.training.fairness import validate_baseline_fairness
from recoalign.training.recoalign_toy import make_toy_batch
from recoalign.training.registry import validate_training_registry
from recoalign.training.sanity import run_training_sanity_suite
from recoalign.training.stages import apply_freeze_policy
from recoalign.training.trainer import ReCoAlignTrainer, TrainingConfig, load_training_config


def training_payload(*, epochs: int = 2) -> dict:
    return {
        "schema_version": 1,
        "training_experiment": "TRAIN001_TEST",
        "stage": "structured_interface_pretraining",
        "model": {
            "name": "recoalign",
            "vision_backbone": "same-vision",
            "llm_backbone": "same-llm",
            "dataset_family": "toy",
            "visual_dim": 8,
            "interface_dim": 16,
            "llm_dim": 12,
            "num_structure_tokens": 4,
            "num_heads": 4,
            "num_object_classes": 2,
            "num_attribute_classes": 2,
            "num_relation_classes": 2,
            "num_composition_classes": 2,
            "num_answer_classes": 2,
        },
        "dataset": {
            "name": "toy",
            "split": "train",
            "manifest": "manifests/datasets/synthetic_world_v2.yaml",
            "supervision": ["object", "attribute", "relation", "composition"],
        },
        "optimizer": {"name": "adamw", "weight_decay": 0.0},
        "scheduler": {"name": "constant", "warmup_steps": 0},
        "learning_rate": 0.005,
        "batch_size": 8,
        "epochs": epochs,
        "freeze_policy": "interface_only",
        "loss_weights": {"semantic": 1.0, "structural": 1.0, "reasoning": 0.0},
        "seed": 7,
        "validation_every": 1,
        "save_every": 1,
        "output_dir": "runs/training-tests",
        "inference_contract": {"oracle_graph_allowed": False},
    }


def batches(config: TrainingConfig) -> tuple[dict, dict]:
    model_config = ReCoAlignConfig.from_mapping(config.raw)
    train = make_toy_batch(batch_size=8, config=model_config, seed=8).as_dict()
    validation = make_toy_batch(batch_size=8, config=model_config, seed=9).as_dict()
    return train, validation


def test_stage_configs_and_training_registry_validate() -> None:
    for path in (
        "configs/training/recoalign_stage1.yaml",
        "configs/training/recoalign_stage2.yaml",
        "configs/training/recoalign_stage3.yaml",
    ):
        config = load_training_config(path)
        assert config.batch_size > 0
        assert config.loss_weights
    assert validate_training_registry()["valid"]


def test_trainer_writes_auditable_artifacts_and_probe_metrics(tmp_path) -> None:
    config = TrainingConfig.from_mapping(training_payload())
    train, validation = batches(config)
    trainer = ReCoAlignTrainer.from_config(
        config,
        run_dir=tmp_path / "run",
        capture_environment_metadata=False,
    )
    result = trainer.fit([train], [validation])
    assert result["loss_decreased"]
    assert result["oracle_graph_used_at_inference"] is False
    assert result["latest_validation"]["structure_probe_accuracy"] >= 0.0
    for relative in (
        "config.resolved.yaml",
        "environment.json",
        "git_commit.txt",
        "dataset_manifest.yaml",
        "checkpoint_manifest.yaml",
        "metrics.json",
        "logs/training.jsonl",
        "logs/validation.jsonl",
        "logs/loss_curves.svg",
    ):
        assert (tmp_path / "run" / relative).is_file()


def test_training_resume_restores_optimizer_scheduler_and_step(tmp_path) -> None:
    config = TrainingConfig.from_mapping(training_payload(epochs=4))
    train, validation = batches(config)
    first = ReCoAlignTrainer.from_config(
        config, run_dir=tmp_path / "resume", capture_environment_metadata=False
    )
    partial = first.fit([train], [validation], stop_after_epoch=2)
    resumed = ReCoAlignTrainer.from_config(
        config, run_dir=tmp_path / "resume", capture_environment_metadata=False
    )
    complete = resumed.fit([train], [validation], resume_from=partial["checkpoint"])
    assert complete["epochs_completed"] == 4
    assert complete["global_step"] == 4
    assert len((tmp_path / "resume" / "logs" / "training.jsonl").read_text().splitlines()) == 4


def test_oracle_graph_is_rejected_as_model_input(tmp_path) -> None:
    config = TrainingConfig.from_mapping(training_payload())
    train, validation = batches(config)
    train["oracle_graph"] = torch.ones(8, 2)
    trainer = ReCoAlignTrainer.from_config(
        config, run_dir=tmp_path / "forbidden", capture_environment_metadata=False
    )
    with pytest.raises(ValueError, match="oracle graph inference input is forbidden"):
        trainer.fit([train], [validation])


class Wrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = nn.Linear(2, 2)
        self.llm = nn.Linear(2, 2)
        self.interface = nn.Linear(2, 2)
        self.projector = nn.Linear(2, 2)


def test_three_freeze_policies_are_distinct() -> None:
    wrapper = Wrapper()
    interface = apply_freeze_policy(wrapper, "interface_only")
    assert interface["trainable_names"] == ["interface.weight", "interface.bias"]
    interface_projector = apply_freeze_policy(wrapper, "interface_and_projector")
    assert "projector.weight" in interface_projector["trainable_names"]
    full = apply_freeze_policy(wrapper, "full_finetuning")
    assert full["frozen_parameter_tensors"] == 0
    assert full["trainable_parameter_tensors"] == 8


def test_ablation_and_random_label_controls() -> None:
    base = training_payload()
    assert apply_ablation(base, "no_structure")["model"]["use_structure_tokens"] is False
    assert load_ablation_config("configs/ablations/no_semantic_loss.yaml")["loss_weights"][
        "semantic"
    ] == 0.0
    config = TrainingConfig.from_mapping(base)
    train, _ = batches(config)
    randomized = randomize_structural_labels(train, seed=11)
    assert any(
        not torch.equal(train[name], randomized[name])
        for name in ("object_targets", "attribute_targets", "relation_targets")
    )


def test_baseline_fairness_rejects_backbone_mismatch() -> None:
    controlled = {
        "vision_backbone": "v",
        "llm_backbone": "l",
        "dataset": "d",
        "split": "test",
        "seed": 1,
        "oracle_graph_at_inference": False,
    }
    matrix = {
        name: deepcopy(controlled)
        for name in ("original_vlm", "caption_adapter", "oracle_graph_prompt", "recoalign")
    }
    matrix["oracle_graph_prompt"]["oracle_graph_at_inference"] = True
    assert validate_baseline_fairness(matrix)["valid"]
    matrix["caption_adapter"]["vision_backbone"] = "different"
    with pytest.raises(ValueError, match="unfair baseline matrix"):
        validate_baseline_fairness(matrix)


def test_controlled_sanity_suite_observes_random_label_decline(tmp_path) -> None:
    payload = training_payload(epochs=30)
    payload["batch_size"] = 100
    result = run_training_sanity_suite(payload, output_dir=tmp_path / "sanity")
    assert result["samples"] == 100
    assert result["overfit"]["passed"]
    assert result["random_label"]["expected_decline_observed"]
    assert result["no_structure"]["analyzable"]
    assert result["resume"]["passed"]
    saved = json.loads((tmp_path / "sanity" / "sanity_report.json").read_text())
    assert saved["oracle_graph_used_at_inference"] is False
