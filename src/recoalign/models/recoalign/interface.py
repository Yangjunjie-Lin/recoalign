"""Composition of the learned structure-token encoder and reasoning adapter."""

from __future__ import annotations

from torch import Tensor, nn

from recoalign.models.interface import ReasoningInterfaceAdapter, StructureTokenEncoder


class ReCoAlignInterface(nn.Module):
    """Minimal trainable interface: visual tokens → structure tokens → context."""

    def __init__(
        self,
        visual_dim: int,
        interface_dim: int,
        llm_dim: int,
        num_structure_tokens: int = 8,
        *,
        num_heads: int = 4,
        dropout: float = 0.0,
        include_visual_tokens: bool = True,
        typed_tokens: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = StructureTokenEncoder(
            visual_dim,
            interface_dim,
            num_structure_tokens,
            num_heads=num_heads,
            dropout=dropout,
            typed=typed_tokens,
        )
        self.adapter = ReasoningInterfaceAdapter(
            visual_dim,
            interface_dim,
            llm_dim,
            include_visual_tokens=include_visual_tokens,
        )

    def extract_structure_tokens(self, visual_tokens: Tensor) -> Tensor:
        return self.encoder(visual_tokens)

    def forward(self, visual_tokens: Tensor) -> tuple[Tensor, Tensor]:
        structure_tokens = self.extract_structure_tokens(visual_tokens)
        context = self.adapter(visual_tokens, structure_tokens)
        return structure_tokens, context


__all__ = ["ReCoAlignInterface"]
