"""Benchmark adapter interfaces and normalized records."""

from recoalign.benchmarks.base import BenchmarkAdapter, BenchmarkResult
from recoalign.benchmarks.records import PairwiseCaptionRecord, RetrievalRecord

__all__ = [
    "BenchmarkAdapter",
    "BenchmarkResult",
    "PairwiseCaptionRecord",
    "RetrievalRecord",
]
