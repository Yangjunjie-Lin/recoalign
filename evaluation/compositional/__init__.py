"""Compositional benchmark adapters shared by every registered BaseVLM."""

from recoalign.evaluation.vlm_benchmark.adapters import (
    BenchmarkDataset,
    BenchmarkSample,
    load_benchmark_dataset,
)

__all__ = ["BenchmarkDataset", "BenchmarkSample", "load_benchmark_dataset"]
