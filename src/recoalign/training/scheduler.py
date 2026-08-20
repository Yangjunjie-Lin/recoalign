"""Minimal, explicit learning-rate schedules."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: str | Mapping[str, Any],
    *,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    payload = {"name": config} if isinstance(config, str) else dict(config)
    name = str(payload.get("name", "constant")).lower()
    warmup_steps = int(payload.get("warmup_steps", 0))
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be non-negative and smaller than total_steps")

    def warmup(step: int) -> float:
        return min(1.0, float(step + 1) / max(1, warmup_steps)) if warmup_steps else 1.0

    if name == "constant":
        rule = warmup
    elif name == "linear":
        def rule(step: int) -> float:
            if step < warmup_steps:
                return warmup(step)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 1.0 - progress)
    elif name == "cosine":
        def rule(step: int) -> float:
            if step < warmup_steps:
                return warmup(step)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    else:
        raise ValueError(f"unsupported scheduler: {name}")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, rule)


__all__ = ["build_scheduler"]
