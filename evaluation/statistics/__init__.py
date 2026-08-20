"""Statistical primitives used by registered ReCoAlign experiments."""

from evaluation.statistics.core import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_SEEDS,
    KEY_EXPERIMENT_SEEDS,
    bootstrap_confidence_interval,
    multiple_seed_summary,
    paired_bootstrap_test,
    paired_t_test,
    permutation_test,
)

__all__ = [
    "DEFAULT_CONFIDENCE_LEVEL",
    "DEFAULT_SEEDS",
    "KEY_EXPERIMENT_SEEDS",
    "bootstrap_confidence_interval",
    "multiple_seed_summary",
    "paired_bootstrap_test",
    "paired_t_test",
    "permutation_test",
]
