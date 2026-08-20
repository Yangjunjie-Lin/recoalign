"""Fixed-encoder and supervision-strength controls."""

from __future__ import annotations

import torch
from torch import Tensor, nn

SUPERVISION_LEVELS = ("full", "weak", "none")


class FixedGraphEncoderControl(nn.Module):
    """Frozen visual-to-slot control with no learned structure-token queries.

    This is a capacity/protocol control, not a claim that a graph is available from
    pixels. It is intentionally frozen and accepts visual features only; an oracle
    graph must be evaluated separately as the external upper bound.
    """

    def __init__(
        self,
        visual_dim: int,
        interface_dim: int,
        num_tokens: int = 8,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if min(visual_dim, interface_dim, num_tokens) <= 0:
            raise ValueError("fixed encoder dimensions must be positive")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        weight = torch.randn(interface_dim, visual_dim, generator=generator) / visual_dim**0.5
        bias = torch.randn(interface_dim, generator=generator) / interface_dim**0.5
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)
        self.num_tokens = int(num_tokens)
        self.visual_dim = int(visual_dim)
        self.interface_dim = int(interface_dim)

    def forward(self, visual_tokens: Tensor) -> Tensor:
        if visual_tokens.ndim != 3 or visual_tokens.shape[-1] != self.visual_dim:
            raise ValueError("visual_tokens do not match fixed encoder dimensions")
        pooled = visual_tokens.mean(dim=1) @ self.weight.t() + self.bias
        return pooled.unsqueeze(1).expand(-1, self.num_tokens, -1).contiguous()


def apply_supervision_ablation(
    batch: dict[str, Tensor | None], level: str
) -> dict[str, Tensor | None]:
    """Return a copy of a batch with preregistered graph-supervision strength."""

    if level not in SUPERVISION_LEVELS:
        raise ValueError(f"supervision level must be one of {SUPERVISION_LEVELS}")
    result = dict(batch)
    if level == "weak":
        for name in ("object_targets", "attribute_targets", "composition_targets"):
            result[name] = None
    elif level == "none":
        for name in (
            "object_targets",
            "attribute_targets",
            "relation_targets",
            "composition_targets",
        ):
            result[name] = None
    result["supervision_level"] = level
    return result


__all__ = ["FixedGraphEncoderControl", "SUPERVISION_LEVELS", "apply_supervision_ablation"]
