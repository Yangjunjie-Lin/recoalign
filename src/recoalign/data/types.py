"""Dataset-neutral records used by benchmark adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageTextSample:
    """One image and its associated positive text descriptions."""

    sample_id: str
    image_path: Path
    captions: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must be non-empty")
        if not self.captions or any(not caption.strip() for caption in self.captions):
            raise ValueError("captions must contain at least one non-empty string")
