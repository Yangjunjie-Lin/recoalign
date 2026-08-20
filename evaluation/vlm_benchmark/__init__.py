"""Compatibility exports for the canonical Phase-4 evaluation package."""

from recoalign.evaluation.vlm_benchmark import (
    BenchmarkDataset,
    BenchmarkSample,
    build_evaluation_matrix,
    load_benchmark_dataset,
    run_benchmark_cell,
    validate_evaluation_matrix,
)

__all__ = [
    "BenchmarkDataset",
    "BenchmarkSample",
    "build_evaluation_matrix",
    "load_benchmark_dataset",
    "run_benchmark_cell",
    "validate_evaluation_matrix",
]
