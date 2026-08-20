"""Config-driven ReCoAlign ablations and controlled toy execution."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml

from .recoalign_toy import make_toy_batch
from .trainer import ReCoAlignTrainer, TrainingConfig

ABLATION_NAMES = (
    "no_structure",
    "random_structure",
    "no_semantic_loss",
    "no_reasoning_loss",
)


def apply_ablation(config: dict[str, Any], ablation: str) -> dict[str, Any]:
    if ablation not in ABLATION_NAMES:
        raise ValueError(f"unsupported ablation: {ablation}")
    resolved = deepcopy(config)
    model = resolved.setdefault("model", {})
    weights = resolved.setdefault("loss_weights", {})
    if ablation == "no_structure":
        model["use_structure_tokens"] = False
        weights["structural"] = 0.0
    elif ablation == "random_structure":
        model["use_structure_tokens"] = True
        model["random_structure_tokens"] = True
        weights["structural"] = 0.0
    elif ablation == "no_semantic_loss":
        weights["semantic"] = 0.0
    elif ablation == "no_reasoning_loss":
        weights["reasoning"] = 0.0
    resolved["ablation"] = ablation
    return resolved


def load_ablation_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("ablation") not in ABLATION_NAMES:
        raise ValueError("ablation config must declare a supported ablation")
    base = payload.get("base_config")
    if not base:
        raise ValueError("ablation config requires base_config")
    base_payload = yaml.safe_load(Path(str(base)).read_text(encoding="utf-8"))
    if not isinstance(base_payload, dict):
        raise ValueError("ablation base config root must be a mapping")
    resolved = apply_ablation(base_payload, str(payload["ablation"]))
    overrides = payload.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("ablation overrides must be a mapping")
    return _deep_update(resolved, overrides)


def run_toy_ablation(
    config: dict[str, Any],
    *,
    run_dir: str | Path,
    capture_environment_metadata: bool = False,
) -> dict[str, Any]:
    training_config = TrainingConfig.from_mapping(config)
    trainer = ReCoAlignTrainer.from_config(
        training_config,
        run_dir=run_dir,
        capture_environment_metadata=capture_environment_metadata,
    )
    batch = make_toy_batch(
        batch_size=training_config.batch_size,
        config=trainer.model.config,
        seed=training_config.seed + 1,
    ).as_dict()
    return trainer.fit([batch], [batch])


def randomize_structural_labels(
    batch: dict[str, torch.Tensor | None], *, seed: int
) -> dict[str, torch.Tensor | None]:
    randomized = dict(batch)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for name in ("object_targets", "attribute_targets", "relation_targets", "composition_targets"):
        target = randomized.get(name)
        if isinstance(target, torch.Tensor):
            randomized[name] = target[torch.randperm(len(target), generator=generator)]
    return randomized


def _deep_update(target: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


__all__ = [
    "ABLATION_NAMES",
    "apply_ablation",
    "load_ablation_config",
    "randomize_structural_labels",
    "run_toy_ablation",
]
