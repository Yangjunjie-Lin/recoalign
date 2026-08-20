"""Numerical token similarity, cluster, and attention summaries."""

from __future__ import annotations

from typing import Any

import numpy as np


def similarity_matrix(values: np.ndarray) -> np.ndarray:
    features = np.asarray(values, dtype=np.float64)
    if features.ndim not in {2, 3}:
        raise ValueError("values must be [samples, dimension] or [samples, tokens, dimension]")
    if features.ndim == 3:
        features = features.mean(axis=1)
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    normalized = features / np.maximum(norms, 1e-12)
    return normalized @ normalized.T


def token_clusters(values: np.ndarray, *, threshold: float = 0.8) -> dict[str, Any]:
    """Build deterministic connected components from cosine similarity."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("cluster threshold must be in [0, 1]")
    matrix = similarity_matrix(values)
    parent = list(range(len(matrix)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(matrix)):
        for right in range(left + 1, len(matrix)):
            if matrix[left, right] >= threshold:
                parent[find(right)] = find(left)
    groups: dict[int, list[int]] = {}
    for index in range(len(matrix)):
        groups.setdefault(find(index), []).append(index)
    return {
        "threshold": threshold,
        "cluster_count": len(groups),
        "clusters": [members for _root, members in sorted(groups.items())],
        "similarity_mean": float(matrix[np.triu_indices_from(matrix, k=1)].mean())
        if len(matrix) > 1
        else 1.0,
    }


def attention_summary(attention: np.ndarray, structure_token_start: int) -> dict[str, Any]:
    values = np.asarray(attention, dtype=np.float64)
    if values.ndim < 2 or not 0 <= structure_token_start < values.shape[-1]:
        raise ValueError("attention must include a valid structure token start index")
    structure = values[..., structure_token_start:]
    visual = values[..., :structure_token_start]
    return {
        "structure_token_start": structure_token_start,
        "mean_structure_attention": float(structure.mean()),
        "mean_visual_attention": float(visual.mean()) if visual.size else 0.0,
        "structure_attention_mass": float(structure.sum(axis=-1).mean()),
        "visual_attention_mass": float(visual.sum(axis=-1).mean()) if visual.size else 0.0,
        "structure_over_visual_ratio": (
            float(structure.mean() / visual.mean()) if visual.size and visual.mean() else None
        ),
    }


__all__ = ["attention_summary", "similarity_matrix", "token_clusters"]
