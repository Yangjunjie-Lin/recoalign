"""Dependency-light probes for representation availability."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence

import numpy as np


def centroid_probe_accuracy(features: np.ndarray, labels: Sequence[Hashable]) -> float:
    """Nearest class-centroid probe used as an availability diagnostic, not a causal claim."""

    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or len(values) != len(labels) or not len(values):
        raise ValueError("features must be a non-empty 2D array aligned with labels")
    centroids: dict[Hashable, np.ndarray] = {}
    grouped: dict[Hashable, list[np.ndarray]] = defaultdict(list)
    for feature, label in zip(values, labels, strict=True):
        grouped[label].append(feature)
    for label, rows in grouped.items():
        centroids[label] = np.mean(rows, axis=0)
    correct = 0
    for feature, label in zip(values, labels, strict=True):
        prediction = min(centroids, key=lambda key: float(np.linalg.norm(feature - centroids[key])))
        correct += int(prediction == label)
    return correct / len(values)
