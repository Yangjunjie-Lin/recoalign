"""Paired seed statistics for ablations and interventions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from evaluation.statistics import paired_bootstrap_test


def summarize_ablation_seeds(
    full_values: Sequence[float],
    ablation_values: Sequence[float],
    *,
    seed_ids: Sequence[int] | None = None,
    samples: int = 10_000,
    seed: int = 7,
) -> dict[str, Any]:
    if len(full_values) != len(ablation_values) or not full_values:
        raise ValueError("full and ablation values must be non-empty and paired")
    if seed_ids is not None and len(seed_ids) != len(full_values):
        raise ValueError("seed_ids must align with ablation values")
    test = paired_bootstrap_test(
        list(full_values), list(ablation_values), samples=samples, seed=seed
    )
    differences = [
        float(left) - float(right)
        for left, right in zip(full_values, ablation_values, strict=True)
    ]
    return {
        "full": _summary(full_values),
        "ablation": _summary(ablation_values),
        "full_minus_ablation": _summary(differences),
        "paired_test": test,
        "seed_ids": list(seed_ids) if seed_ids is not None else None,
        "n_seeds": len(full_values),
    }


def _summary(values: Sequence[float]) -> dict[str, Any]:
    numeric = [float(value) for value in values]
    mean = sum(numeric) / len(numeric)
    variance = sum((value - mean) ** 2 for value in numeric) / max(1, len(numeric) - 1)
    return {"mean": mean, "std": variance**0.5, "values": numeric}


__all__ = ["summarize_ablation_seeds"]
