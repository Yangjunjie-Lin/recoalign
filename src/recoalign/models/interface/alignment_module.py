"""Semantic alignment utilities for the visual/structure interface."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class StructureAlignmentModule(nn.Module):
    """Align pooled visual and structure summaries in the interface space."""

    def __init__(self, visual_dim: int, interface_dim: int) -> None:
        super().__init__()
        if min(visual_dim, interface_dim) <= 0:
            raise ValueError("alignment dimensions must be positive")
        self.visual_projection = nn.Linear(visual_dim, interface_dim)
        self.structure_projection = nn.Linear(interface_dim, interface_dim)

    def forward(self, visual_tokens: Tensor, structure_tokens: Tensor) -> tuple[Tensor, Tensor]:
        if visual_tokens.ndim != 3 or structure_tokens.ndim != 3:
            raise ValueError("visual_tokens and structure_tokens must be rank-3 tensors")
        visual_summary = self.visual_projection(visual_tokens.mean(dim=1))
        structure_summary = self.structure_projection(structure_tokens.mean(dim=1))
        return visual_summary, structure_summary

    def cosine_similarity(self, visual_tokens: Tensor, structure_tokens: Tensor) -> Tensor:
        visual_summary, structure_summary = self(visual_tokens, structure_tokens)
        return torch.nn.functional.cosine_similarity(visual_summary, structure_summary, dim=-1)


__all__ = ["StructureAlignmentModule"]
