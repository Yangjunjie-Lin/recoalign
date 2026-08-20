"""Public structure-token extraction boundary."""

from __future__ import annotations

from torch import Tensor, nn

from recoalign.models.interface import StructureTokenEncoder


class StructureTokenizer(nn.Module):
    """A named facade for extracting latent tokens from frozen or trainable features."""

    def __init__(self, visual_dim: int, interface_dim: int, num_tokens: int = 8, **kwargs) -> None:
        super().__init__()
        self.encoder = StructureTokenEncoder(
            visual_dim,
            interface_dim,
            num_tokens,
            **kwargs,
        )

    def forward(self, visual_tokens: Tensor) -> Tensor:
        return self.encoder(visual_tokens)

    def extract(self, visual_tokens: Tensor) -> Tensor:
        return self.forward(visual_tokens)


__all__ = ["StructureTokenizer"]
