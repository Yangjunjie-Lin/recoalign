from __future__ import annotations

from typing import Any

import numpy as np


def _summary(samples: np.ndarray) -> dict[str, float]:
    return {
        "bootstrap_mean": float(np.mean(samples)),
        "bootstrap_std": float(np.std(samples, ddof=1)),
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
    }


def bootstrap_accuracy(correct: np.ndarray, *, samples: int, seed: int) -> dict[str, float]:
    correct = np.asarray(correct, dtype=np.float64)
    if correct.ndim != 1 or correct.size == 0:
        raise ValueError("correct must be a non-empty 1D array")
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = rng.integers(0, correct.size, size=correct.size)
        values[index] = float(np.mean(correct[draw]))
    return _summary(values)


def bootstrap_drop(
    correct_zv: np.ndarray,
    correct_za: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    correct_zv = np.asarray(correct_zv, dtype=np.float64)
    correct_za = np.asarray(correct_za, dtype=np.float64)
    if correct_zv.shape != correct_za.shape or correct_zv.ndim != 1:
        raise ValueError("Zv and Za correctness arrays must be paired 1D arrays")
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = rng.integers(0, correct_zv.size, size=correct_zv.size)
        values[index] = float(np.mean(correct_zv[draw]) - np.mean(correct_za[draw]))
    return {"drop": float(np.mean(correct_zv) - np.mean(correct_za)), **_summary(values)}


def bootstrap_drop_comparison(
    *,
    object_zv: np.ndarray,
    object_za: np.ndarray,
    relation_zv: np.ndarray,
    relation_za: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (object_zv, object_za, relation_zv, relation_za)
    ]
    if any(value.ndim != 1 for value in arrays) or len({value.shape for value in arrays}) != 1:
        raise ValueError("All correctness arrays must be paired and have identical 1D shapes")
    object_zv, object_za, relation_zv, relation_za = arrays
    observed = float(
        (np.mean(relation_zv) - np.mean(relation_za)) - (np.mean(object_zv) - np.mean(object_za))
    )
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = rng.integers(0, object_zv.size, size=object_zv.size)
        object_drop = np.mean(object_zv[draw]) - np.mean(object_za[draw])
        relation_drop = np.mean(relation_zv[draw]) - np.mean(relation_za[draw])
        values[index] = relation_drop - object_drop
    # One-sided test of H0: relation drop <= object drop. The +1 correction
    # prevents a spuriously exact zero p-value at finite bootstrap count.
    p_one_sided = float((np.count_nonzero(values <= 0.0) + 1) / (samples + 1))
    lower_tail = (np.count_nonzero(values <= 0.0) + 1) / (samples + 1)
    upper_tail = (np.count_nonzero(values >= 0.0) + 1) / (samples + 1)
    p_two_sided = float(min(1.0, 2.0 * min(lower_tail, upper_tail)))
    return {
        "comparison": "relation_drop_minus_object_drop",
        "observed_difference": observed,
        **_summary(values),
        "p_one_sided": p_one_sided,
        "p_two_sided": p_two_sided,
        "bootstrap_samples": samples,
    }
