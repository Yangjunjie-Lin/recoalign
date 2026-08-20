"""Adapter from latent structure tokens to a language-model hidden space."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ReasoningInterfaceAdapter(nn.Module):
    """Create a gated multimodal context without using an oracle graph.

    Visual tokens remain available to preserve appearance semantics.  Structure
    tokens are projected to the language hidden dimension and gated per slot; the
    resulting sequence can be passed to a VLM/LLM as its multimodal context.
    """

    def __init__(
        self,
        visual_dim: int,
        interface_dim: int,
        llm_dim: int,
        *,
        include_visual_tokens: bool = True,
    ) -> None:
        super().__init__()
        if min(visual_dim, interface_dim, llm_dim) <= 0:
            raise ValueError("adapter dimensions must be positive")
        self.visual_dim = int(visual_dim)
        self.interface_dim = int(interface_dim)
        self.llm_dim = int(llm_dim)
        self.include_visual_tokens = bool(include_visual_tokens)
        self.visual_projection = nn.Linear(visual_dim, llm_dim)
        self.structure_projection = nn.Linear(interface_dim, llm_dim)
        self.gate = nn.Sequential(
            nn.Linear(llm_dim * 2, llm_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(llm_dim)

    def forward(self, visual_tokens: Tensor, structure_tokens: Tensor) -> Tensor:
        if visual_tokens.ndim != 3 or structure_tokens.ndim != 3:
            raise ValueError("visual_tokens and structure_tokens must be rank-3 tensors")
        if visual_tokens.shape[0] != structure_tokens.shape[0]:
            raise ValueError("visual and structure token batches must match")
        visual_context = self.visual_projection(visual_tokens)
        structure_context = self.structure_projection(structure_tokens)
        visual_summary = visual_context.mean(dim=1, keepdim=True)
        gate_input = torch.cat(
            (structure_context, visual_summary.expand(-1, structure_context.shape[1], -1)),
            dim=-1,
        )
        gated_structure = structure_context * self.gate(gate_input)
        if self.include_visual_tokens:
            context = torch.cat((visual_context, gated_structure), dim=1)
        else:
            context = gated_structure
        return self.norm(context)

    def adapt(self, visual_tokens: Tensor, structure_tokens: Tensor) -> Tensor:
        """Named alias useful to adapters that follow an ``align/prepare`` API."""

        return self(visual_tokens, structure_tokens)


__all__ = ["ReasoningInterfaceAdapter"]
