"""Controlled optimizer construction for ReCoAlign training."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn


def build_optimizer(
    model: nn.Module,
    config: str | Mapping[str, Any],
    *,
    learning_rate: float,
) -> torch.optim.Optimizer:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    payload = {"name": config} if isinstance(config, str) else dict(config)
    name = str(payload.get("name", "adamw")).lower()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("optimizer cannot be built without trainable parameters")
    weight_decay = float(payload.get("weight_decay", 0.0))
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            betas=_betas(payload),
            weight_decay=weight_decay,
        )
    if name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            betas=_betas(payload),
            weight_decay=weight_decay,
        )
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=float(payload.get("momentum", 0.0)),
            weight_decay=weight_decay,
        )
    raise ValueError(f"unsupported optimizer: {name}")


def optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}


def _betas(payload: Mapping[str, Any]) -> tuple[float, float]:
    raw = payload.get("betas", (0.9, 0.999))
    if not isinstance(raw, Iterable):
        raise ValueError("optimizer betas must contain two numbers")
    values = tuple(float(value) for value in raw)
    if len(values) != 2 or any(not 0 <= value < 1 for value in values):
        raise ValueError("optimizer betas must contain two values in [0, 1)")
    return values


__all__ = ["build_optimizer", "optimizer_parameter_ids"]
