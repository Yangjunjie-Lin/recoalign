"""Dataset-agnostic image-text retrieval metrics."""

from collections.abc import Iterable

import numpy as np


def recall_at_k(
    similarity: np.ndarray,
    positives: np.ndarray,
    ks: Iterable[int] = (1, 5, 10),
) -> dict[int, float]:
    """Compute query-level recall for one retrieval direction.

    Args:
        similarity: Matrix with shape ``[num_queries, num_candidates]``. Larger is better.
        positives: Boolean matrix with the same shape. ``positives[q, c]`` is true when
            candidate ``c`` is relevant to query ``q``.
        ks: Retrieval cutoffs.

    Returns:
        Mapping from each cutoff to recall as a percentage in ``[0, 100]``.
    """
    similarity = np.asarray(similarity)
    positives = np.asarray(positives, dtype=bool)

    if similarity.ndim != 2:
        raise ValueError("similarity must be a two-dimensional matrix")
    if positives.shape != similarity.shape:
        raise ValueError("positives must have the same shape as similarity")
    if similarity.shape[0] == 0 or similarity.shape[1] == 0:
        raise ValueError("retrieval matrices must be non-empty")
    if not np.all(positives.any(axis=1)):
        raise ValueError("every query must have at least one positive candidate")

    cutoffs = tuple(sorted(set(int(k) for k in ks)))
    if not cutoffs or cutoffs[0] <= 0:
        raise ValueError("all recall cutoffs must be positive integers")

    ranking = np.argsort(-similarity, axis=1)
    metrics: dict[int, float] = {}
    for k in cutoffs:
        effective_k = min(k, similarity.shape[1])
        top_k = ranking[:, :effective_k]
        hits = np.take_along_axis(positives, top_k, axis=1).any(axis=1)
        metrics[k] = float(hits.mean() * 100.0)
    return metrics


def summarize_bidirectional_retrieval(
    similarity: np.ndarray,
    image_to_text_positives: np.ndarray,
    ks: Iterable[int] = (1, 5, 10),
) -> dict[str, float]:
    """Compute image-to-text, text-to-image, and mean recall metrics.

    ``image_to_text_positives`` has one row per image and one column per caption.
    The transposed matrix defines positives for text-to-image retrieval.
    """
    similarity = np.asarray(similarity)
    positives = np.asarray(image_to_text_positives, dtype=bool)

    i2t = recall_at_k(similarity, positives, ks)
    t2i = recall_at_k(similarity.T, positives.T, ks)

    summary: dict[str, float] = {}
    for k, value in i2t.items():
        summary[f"i2t_R@{k}"] = value
    for k, value in t2i.items():
        summary[f"t2i_R@{k}"] = value

    summary["mean_recall"] = float(np.mean([*i2t.values(), *t2i.values()]))
    return summary
