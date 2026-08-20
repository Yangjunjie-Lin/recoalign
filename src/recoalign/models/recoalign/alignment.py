"""ReCoAlign alignment boundary kept separate for ablations and instrumentation."""

from __future__ import annotations

from torch import Tensor, nn

from recoalign.models.interface import StructureAlignmentModule


class ReCoAlignAlignment(nn.Module):
    """Return aligned visual/structure summaries for semantic preservation."""

    def __init__(self, visual_dim: int, interface_dim: int) -> None:
        super().__init__()
        self.module = StructureAlignmentModule(visual_dim, interface_dim)

    def forward(self, visual_tokens: Tensor, structure_tokens: Tensor) -> tuple[Tensor, Tensor]:
        return self.module(visual_tokens, structure_tokens)


__all__ = ["ReCoAlignAlignment"]
