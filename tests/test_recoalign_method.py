from __future__ import annotations

import torch

from recoalign.models.recoalign import LOSS_NAMES, ReCoAlignConfig, ReCoAlignModel
from recoalign.training.recoalign_toy import (
    checkpoint_roundtrip,
    load_recoalign_config,
    make_toy_batch,
    overfit_sanity_test,
    run_toy_training,
)


def small_config(**overrides) -> ReCoAlignConfig:
    values = {
        "visual_dim": 8,
        "interface_dim": 16,
        "llm_dim": 12,
        "num_structure_tokens": 4,
        "num_heads": 4,
        "num_relation_classes": 3,
        "num_answer_classes": 2,
    }
    values.update(overrides)
    return ReCoAlignConfig(**values)


def test_structure_token_extraction_shape_and_no_oracle_input() -> None:
    model = ReCoAlignModel(small_config())
    visual = torch.randn(3, 5, 8)
    tokens = model.extract_structure_tokens(visual)
    assert tokens.shape == (3, 4, 16)
    assert not hasattr(model.forward, "graph")


def test_forward_exposes_reasoning_context_and_three_loss_terms() -> None:
    model = ReCoAlignModel(small_config())
    batch = make_toy_batch(batch_size=3, visual_tokens=5, config=model.config, seed=2)
    outputs = model(batch.visual_tokens)
    terms = model.compute_loss(
        outputs,
        semantic_targets=batch.semantic_targets,
        relation_targets=batch.relation_targets,
        answer_labels=batch.answer_labels,
    )
    assert outputs["context"].shape[-1] == 12
    assert set(LOSS_NAMES).issubset(terms)
    assert len(model.loss_names) == 3
    assert torch.isfinite(terms["total"])


def test_toy_training_loss_decreases() -> None:
    result = run_toy_training(small_config(), steps=8, batch_size=4, seed=3)
    assert result["loss_decreased"]
    assert result["final_loss"] < result["initial_loss"]


def test_tiny_batch_overfit_sanity() -> None:
    result = overfit_sanity_test(small_config(), steps=35, seed=4)
    assert result["loss_decreased"]
    assert result["overfit_pass"]


def test_checkpoint_roundtrip_preserves_structure_tokens(tmp_path) -> None:
    model = ReCoAlignModel(small_config())
    visual = torch.randn(2, 5, 8)
    result = checkpoint_roundtrip(model, visual, tmp_path / "recoalign.pt")
    assert result["identical"]
    assert result["max_abs_difference"] == 0.0


def test_no_structure_and_random_structure_ablation_boundaries() -> None:
    visual = torch.randn(2, 5, 8)
    no_structure = ReCoAlignModel(small_config(use_structure_tokens=False))
    assert torch.count_nonzero(no_structure.extract_structure_tokens(visual)) == 0
    random_structure = ReCoAlignModel(small_config(random_structure_tokens=True))
    first = random_structure.extract_structure_tokens(visual)
    second = random_structure.extract_structure_tokens(visual)
    assert first.shape == second.shape
    assert not torch.equal(first, second)


def test_yaml_config_and_cli_contract() -> None:
    config = load_recoalign_config("configs/models/recoalign.yaml")
    assert config.num_structure_tokens == 8
    assert config.to_dict()["structural_loss_weight"] == 1.0
