"""Metrics for image with positive-caption versus hard-negative-caption evaluation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairwiseResult:
    positive_scores: np.ndarray
    negative_scores: np.ndarray
    correct: np.ndarray
    ties: np.ndarray


def evaluate_pairwise_scores(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
) -> PairwiseResult:
    """Apply the strict SugarCrepe decision rule; ties count as incorrect."""
    positive = np.asarray(positive_scores, dtype=np.float32)
    negative = np.asarray(negative_scores, dtype=np.float32)
    if positive.ndim != 1 or negative.shape != positive.shape or positive.size == 0:
        raise ValueError("positive_scores and negative_scores must be equal non-empty vectors")
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError("pairwise scores must be finite")
    return PairwiseResult(
        positive_scores=positive,
        negative_scores=negative,
        correct=positive > negative,
        ties=positive == negative,
    )


def summarize_pairwise(result: PairwiseResult, categories: Sequence[str]) -> dict[str, float]:
    """Report weighted, macro-category, per-category, and tie metrics."""
    if len(categories) != result.correct.size:
        raise ValueError("categories must contain one value per pairwise sample")
    grouped: dict[str, list[bool]] = defaultdict(list)
    for category, correct in zip(categories, result.correct, strict=True):
        if not isinstance(category, str) or not category.strip():
            raise ValueError("categories must be non-empty strings")
        grouped[category].append(bool(correct))
    per_category = {
        category: float(np.mean(values) * 100.0)
        for category, values in sorted(grouped.items())
    }
    metrics = {
        "accuracy": float(np.mean(result.correct) * 100.0),
        "macro_accuracy": float(np.mean(list(per_category.values()))),
        "tie_rate": float(np.mean(result.ties) * 100.0),
    }
    metrics.update({f"accuracy/{category}": value for category, value in per_category.items()})
    return metrics
