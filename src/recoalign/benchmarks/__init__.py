"""Benchmark adapter interfaces and normalized records."""

from recoalign.benchmarks.base import BenchmarkAdapter, BenchmarkResult
from recoalign.benchmarks.records import (
    MultiChoiceRecord,
    PairedMatrixRecord,
    PairwiseCaptionRecord,
    RetrievalRecord,
)

__all__ = [
    "BenchmarkAdapter",
    "BenchmarkResult",
    "MultiChoiceRecord",
    "PairedMatrixRecord",
    "PairwiseCaptionRecord",
    "RetrievalRecord",
]
