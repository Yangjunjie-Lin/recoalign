"""CPU-friendly toy training and sanity checks for the ReCoAlign interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from recoalign.models.recoalign import ReCoAlignConfig, ReCoAlignModel


@dataclass(frozen=True)
class ToyBatch:
    visual_tokens: Tensor
    semantic_targets: Tensor
    relation_targets: Tensor
    answer_labels: Tensor
    object_targets: Tensor | None = None
    attribute_targets: Tensor | None = None
    composition_targets: Tensor | None = None

    def as_dict(self) -> dict[str, Tensor]:
        return {
            "visual_tokens": self.visual_tokens,
            "semantic_targets": self.semantic_targets,
            "relation_targets": self.relation_targets,
            "answer_labels": self.answer_labels,
            "object_targets": self.object_targets,
            "attribute_targets": self.attribute_targets,
            "composition_targets": self.composition_targets,
        }


def make_toy_batch(
    *,
    batch_size: int = 8,
    visual_tokens: int = 6,
    config: ReCoAlignConfig | None = None,
    visual_dim: int | None = None,
    seed: int = 0,
) -> ToyBatch:
    """Generate deterministic visual features with compositional labels.

    The labels are generated from different token positions, so a model that only
    memorizes a pooled caption cannot solve all targets.  No graph object or graph
    token is passed to the model.
    """

    cfg = config or ReCoAlignConfig(visual_dim=visual_dim or 16)
    if batch_size <= 0 or visual_tokens <= 0:
        raise ValueError("batch_size and visual_tokens must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    features = torch.randn(
        batch_size,
        visual_tokens,
        cfg.visual_dim,
        generator=generator,
    )
    pooled = features.mean(dim=1)
    semantic_targets = pooled.new_zeros(batch_size, cfg.interface_dim)
    width = min(cfg.visual_dim, cfg.interface_dim)
    semantic_targets[:, :width] = pooled[:, :width]
    relation_signal = features[:, 0, 0] + features[:, min(1, visual_tokens - 1), 1 % cfg.visual_dim]
    answer_signal = features[:, min(2, visual_tokens - 1), 2 % cfg.visual_dim] - features[
        :, 0, 3 % cfg.visual_dim
    ]
    relation_targets = (relation_signal > 0).long() % cfg.num_relation_classes
    answer_labels = (answer_signal > 0).long() % cfg.num_answer_classes
    object_targets = (features[:, 0, 0] > 0).long() % cfg.num_object_classes
    attribute_targets = (features[:, 0, 1 % cfg.visual_dim] > 0).long()
    attribute_targets %= cfg.num_attribute_classes
    composition_targets = (relation_targets + answer_labels) % cfg.num_composition_classes
    return ToyBatch(
        features,
        semantic_targets,
        relation_targets,
        answer_labels,
        object_targets,
        attribute_targets,
        composition_targets,
    )


def run_toy_training(
    config: ReCoAlignConfig | dict[str, Any] | None = None,
    *,
    steps: int = 30,
    batch_size: int = 8,
    visual_tokens: int = 6,
    seed: int = 0,
    learning_rate: float = 3e-3,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train on one fixed synthetic batch and return auditable loss history."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    torch.manual_seed(seed)
    model = ReCoAlignModel(config)
    model.train()
    batch = make_toy_batch(
        batch_size=batch_size,
        visual_tokens=visual_tokens,
        config=model.config,
        seed=seed + 1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[float] = []
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        terms = model.loss(
            batch.visual_tokens,
            semantic_targets=batch.semantic_targets,
            object_targets=batch.object_targets,
            attribute_targets=batch.attribute_targets,
            relation_targets=batch.relation_targets,
            composition_targets=batch.composition_targets,
            answer_labels=batch.answer_labels,
        )
        terms["total"].backward()
        optimizer.step()
        history.append(float(terms["total"].detach().cpu()))
    checkpoint = None
    if checkpoint_path is not None:
        checkpoint = str(model.save_checkpoint(checkpoint_path, optimizer=optimizer, step=steps))
    return {
        "model": model,
        "batch": batch,
        "loss_history": history,
        "initial_loss": history[0],
        "final_loss": history[-1],
        "loss_decreased": history[-1] < history[0],
        "steps": steps,
        "seed": seed,
        "checkpoint": checkpoint,
    }


def overfit_sanity_test(
    config: ReCoAlignConfig | dict[str, Any] | None = None,
    *,
    steps: int = 100,
    seed: int = 0,
) -> dict[str, Any]:
    """Run a tiny-batch overfit check used before expensive VLM training."""

    result = run_toy_training(config, steps=steps, batch_size=2, seed=seed)
    return {
        "initial_loss": result["initial_loss"],
        "final_loss": result["final_loss"],
        "loss_decreased": result["loss_decreased"],
        "overfit_pass": result["final_loss"] < result["initial_loss"] * 0.8,
    }


def checkpoint_roundtrip(
    model: ReCoAlignModel,
    visual_tokens: Tensor,
    path: str | Path,
) -> dict[str, Any]:
    """Save/load a model and report the maximum output discrepancy."""

    model.eval()
    with torch.no_grad():
        before = model(visual_tokens)["structure_tokens"].detach().cpu()
    model.save_checkpoint(path)
    restored = ReCoAlignModel.from_checkpoint(path)
    restored.eval()
    with torch.no_grad():
        after = restored(visual_tokens)["structure_tokens"].detach().cpu()
    max_difference = float((before - after).abs().max())
    return {
        "path": str(path),
        "max_abs_difference": max_difference,
        "identical": max_difference == 0.0,
    }


def load_recoalign_config(path: str | Path) -> ReCoAlignConfig:
    """Load the method section from a YAML configuration."""

    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ReCoAlign config root must be a mapping")
    return ReCoAlignConfig.from_mapping(payload)


__all__ = [
    "ToyBatch",
    "checkpoint_roundtrip",
    "load_recoalign_config",
    "make_toy_batch",
    "overfit_sanity_test",
    "run_toy_training",
]
