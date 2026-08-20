"""Learnable typed latent slots used as the structured reasoning interface."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class StructureTokenBank(nn.Module):
    """Cross-attend a small bank of learnable structure queries to visual tokens.

    The slots are deliberately latent.  Their optional type embeddings provide a
    weak inductive bias (object/attribute/relation/composition) without imposing a
    hand-written scene graph or a fixed node ordering.
    """

    DEFAULT_TYPES = ("object", "attribute", "relation", "composition")

    def __init__(
        self,
        interface_dim: int,
        num_tokens: int = 8,
        *,
        num_heads: int = 4,
        dropout: float = 0.0,
        typed: bool = True,
        type_names: tuple[str, ...] = DEFAULT_TYPES,
    ) -> None:
        super().__init__()
        if interface_dim <= 0 or num_tokens <= 0:
            raise ValueError("interface_dim and num_tokens must be positive")
        if num_heads <= 0 or interface_dim % num_heads:
            raise ValueError("interface_dim must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not type_names:
            raise ValueError("type_names must not be empty")

        self.interface_dim = int(interface_dim)
        self.num_tokens = int(num_tokens)
        self.typed = bool(typed)
        self.type_names = tuple(type_names)
        self.query = nn.Parameter(torch.empty(num_tokens, interface_dim))
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        self.type_embedding = nn.Embedding(len(self.type_names), interface_dim)
        type_ids = torch.arange(num_tokens) % len(self.type_names)
        self.register_buffer("type_ids", type_ids, persistent=False)
        self.cross_attention = nn.MultiheadAttention(
            interface_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(interface_dim)
        self.norm2 = nn.LayerNorm(interface_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(interface_dim, interface_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(interface_dim * 2, interface_dim),
        )

    def forward(self, visual_tokens: Tensor) -> Tensor:
        """Return ``[batch, num_structure_tokens, interface_dim]`` slots."""

        if visual_tokens.ndim != 3:
            raise ValueError("visual_tokens must have shape [batch, tokens, dimension]")
        batch = visual_tokens.shape[0]
        queries = self.query.unsqueeze(0).expand(batch, -1, -1)
        if self.typed:
            queries = queries + self.type_embedding(self.type_ids).unsqueeze(0)
        attended, _ = self.cross_attention(queries, visual_tokens, visual_tokens)
        slots = self.norm1(queries + attended)
        return self.norm2(slots + self.feed_forward(slots))

    def type_layout(self) -> list[str]:
        """Return the stable diagnostic label assigned to each latent slot."""

        return [self.type_names[int(index)] for index in self.type_ids]


__all__ = ["StructureTokenBank"]
