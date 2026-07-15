"""Model-neutral image and text encoder protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class VisionLanguageEncoder(Protocol):
    """Minimal contract required by benchmark adapters."""

    def encode_images(self, images: Sequence[Any]) -> Any:
        """Return one embedding per image."""
        ...

    def encode_texts(self, texts: Sequence[str]) -> Any:
        """Return one embedding per text."""
        ...
