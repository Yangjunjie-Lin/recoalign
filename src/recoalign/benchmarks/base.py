"""Common interface for retrieval and compositional benchmark adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class BenchmarkResult:
    """Normalized output returned by every benchmark adapter."""

    benchmark: str
    split: str
    metrics: dict[str, float]
    num_samples: int
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkAdapter(Protocol):
    """Protocol implemented by Flickr30K, SugarCrepe, and later adapters."""

    name: str

    def evaluate(self, encoder: Any) -> BenchmarkResult:
        """Evaluate an encoder and return normalized metrics."""
        ...
