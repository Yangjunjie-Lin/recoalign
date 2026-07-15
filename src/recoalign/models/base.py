"""Model-neutral image and text encoder protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np


class VisionLanguageEncoder(Protocol):
    """Minimal contract required by reproducible benchmark evaluators."""

    @property
    def fingerprint(self) -> dict[str, Any]:
        """Return a JSON-serializable identity used for cache and provenance keys."""
        ...

    def encode_image_paths(self, paths: list[Path], *, batch_size: int) -> np.ndarray:
        """Return one normalized float32 embedding per image path."""
        ...

    def encode_texts(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        """Return one normalized float32 embedding per text in input order."""
        ...
