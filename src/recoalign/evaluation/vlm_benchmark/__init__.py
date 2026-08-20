"""Unified BaseVLM benchmark evaluation for Phase 4."""

from .adapters import BenchmarkDataset, BenchmarkSample, load_benchmark_dataset
from .matrix import build_evaluation_matrix, validate_evaluation_matrix
from .runner import run_benchmark_cell

__all__ = [
    "BenchmarkDataset",
    "BenchmarkSample",
    "build_evaluation_matrix",
    "load_benchmark_dataset",
    "run_benchmark_cell",
    "validate_evaluation_matrix",
]
