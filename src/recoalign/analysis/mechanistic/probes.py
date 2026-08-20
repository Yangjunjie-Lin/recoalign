"""Linear probes and before/after representation comparisons."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from diagnosis.interface_diagnosis.probes import linear_probe


def probe_structure_tokens(
    structure_tokens: np.ndarray,
    labels: dict[str, Sequence[object]],
    *,
    seed: int = 7,
) -> dict[str, Any]:
    """Fit the same deterministic linear-probe protocol to each semantic factor."""

    values = np.asarray(structure_tokens, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("structure_tokens must have shape [samples, slots, dimension]")
    pooled = values.mean(axis=1)
    probes = {
        name: linear_probe(pooled, target, seed=seed + index)
        for index, (name, target) in enumerate(sorted(labels.items()))
    }
    available = [row["accuracy"] for row in probes.values() if row.get("status") == "available"]
    return {
        "probes": probes,
        "mean_accuracy": sum(available) / len(available) if available else None,
        "representation": "structure_tokens",
        "seed": seed,
    }


def representation_comparison(
    visual_tokens: np.ndarray,
    structure_tokens: np.ndarray,
    labels: dict[str, Sequence[object]],
    *,
    seed: int = 7,
) -> dict[str, Any]:
    """Compare semantic linear probes before and after the learned interface."""

    visual = np.asarray(visual_tokens, dtype=np.float64)
    structure = np.asarray(structure_tokens, dtype=np.float64)
    if visual.ndim != 3 or structure.ndim != 3 or visual.shape[0] != structure.shape[0]:
        raise ValueError("visual and structure tokens must be aligned rank-3 arrays")
    before = probe_structure_tokens(visual, labels, seed=seed)
    after = probe_structure_tokens(structure, labels, seed=seed)
    per_label: dict[str, Any] = {}
    for name in labels:
        left = before["probes"][name]
        right = after["probes"][name]
        per_label[name] = {
            "visual_accuracy": left.get("accuracy"),
            "structure_accuracy": right.get("accuracy"),
            "delta": (
                right["accuracy"] - left["accuracy"]
                if left.get("accuracy") is not None and right.get("accuracy") is not None
                else None
            ),
        }
    return {
        "before": before,
        "after": after,
        "per_label": per_label,
        "mean_delta": (
            after["mean_accuracy"] - before["mean_accuracy"]
            if after["mean_accuracy"] is not None and before["mean_accuracy"] is not None
            else None
        ),
    }


__all__ = ["probe_structure_tokens", "representation_comparison"]
