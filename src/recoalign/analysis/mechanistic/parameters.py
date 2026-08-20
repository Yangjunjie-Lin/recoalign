"""Parameter-count audits and matched random-capacity controls."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def parameter_audit(model: nn.Module) -> dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    by_module: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        module = name.split(".", 1)[0]
        by_module[module] = by_module.get(module, 0) + parameter.numel()
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": total - trainable,
        "by_module": dict(sorted(by_module.items())),
    }


class ParameterMatchedRandomControl(nn.Module):
    """Equal-capacity random projector used only as a parameter-count control."""

    def __init__(self, input_dim: int, output_dim: int, parameter_count: int) -> None:
        super().__init__()
        if min(input_dim, output_dim, parameter_count) <= 0:
            raise ValueError("parameter-matched control dimensions must be positive")
        self.projector = nn.Linear(input_dim, output_dim)
        extra = parameter_count - sum(
            parameter.numel() for parameter in self.projector.parameters()
        )
        if extra < 0:
            raise ValueError("parameter_count is smaller than the required projector")
        self.random_parameters = nn.Parameter(torch.zeros(extra))
        nn.init.normal_(self.random_parameters, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.projector(values)


def parameter_matched_control(
    model: nn.Module,
    *,
    input_dim: int,
    output_dim: int,
) -> tuple[ParameterMatchedRandomControl, dict[str, Any]]:
    target = sum(parameter.numel() for parameter in model.parameters())
    control = ParameterMatchedRandomControl(input_dim, output_dim, target)
    audit = parameter_audit(control)
    audit["target_parameter_count"] = target
    audit["absolute_count_difference"] = abs(audit["total_parameters"] - target)
    return control, audit


__all__ = ["ParameterMatchedRandomControl", "parameter_audit", "parameter_matched_control"]
