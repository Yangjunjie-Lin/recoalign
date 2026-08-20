"""Visual-token encoder for the learned structured intermediate representation."""

from __future__ import annotations

from torch import Tensor, nn

from .structure_tokens import StructureTokenBank


class StructureTokenEncoder(nn.Module):
    """Project visual tokens and infer a fixed set of latent structure tokens."""

    def __init__(
        self,
        visual_dim: int,
        interface_dim: int,
        num_tokens: int = 8,
        *,
        num_heads: int = 4,
        dropout: float = 0.0,
        typed: bool = True,
    ) -> None:
        super().__init__()
        if visual_dim <= 0:
            raise ValueError("visual_dim must be positive")
        self.visual_dim = int(visual_dim)
        self.interface_dim = int(interface_dim)
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, interface_dim),
        )
        self.token_bank = StructureTokenBank(
            interface_dim,
            num_tokens,
            num_heads=num_heads,
            dropout=dropout,
            typed=typed,
        )

    @property
    def num_tokens(self) -> int:
        return self.token_bank.num_tokens

    def project_visual(self, visual_tokens: Tensor) -> Tensor:
        if visual_tokens.ndim != 3 or visual_tokens.shape[-1] != self.visual_dim:
            raise ValueError(
                "visual_tokens must have shape [batch, tokens, visual_dim] "
                f"with visual_dim={self.visual_dim}"
            )
        return self.visual_projection(visual_tokens)

    def forward(self, visual_tokens: Tensor) -> Tensor:
        return self.token_bank(self.project_visual(visual_tokens))


__all__ = ["StructureTokenEncoder"]
