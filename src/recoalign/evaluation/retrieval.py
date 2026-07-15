"""Dataset-agnostic image-text retrieval metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RankingResult:
    """Per-query retrieval diagnostics without materializing a full similarity matrix."""

    ranks: np.ndarray
    top1_indices: np.ndarray
    top1_scores: np.ndarray
    best_positive_indices: np.ndarray
    best_positive_scores: np.ndarray


def recall_at_k(
    similarity: np.ndarray,
    positives: np.ndarray,
    ks: Iterable[int] = (1, 5, 10),
) -> dict[int, float]:
    """Compute query-level recall for one retrieval direction."""
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

    cutoffs = _normalize_cutoffs(ks)
    ranking = np.argsort(-similarity, axis=1, kind="stable")
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
    """Compute image-to-text, text-to-image, and mean recall metrics."""
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


def rank_queries(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    positive_indices: Sequence[Sequence[int]],
    *,
    batch_size: int = 128,
) -> RankingResult:
    """Rank the best positive for each query in bounded-memory score blocks.

    Ranking matches a stable descending sort: equal scores are ordered by candidate index.
    """
    queries = _validated_embeddings(query_embeddings, "query_embeddings")
    candidates = _validated_embeddings(candidate_embeddings, "candidate_embeddings")
    if queries.shape[1] != candidates.shape[1]:
        raise ValueError("query and candidate embedding dimensions must match")
    if len(positive_indices) != queries.shape[0]:
        raise ValueError("positive_indices must contain one entry per query")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    normalized_positives: list[np.ndarray] = []
    for query_index, indices in enumerate(positive_indices):
        values = np.asarray(list(indices), dtype=np.int64)
        if values.size == 0:
            raise ValueError(f"query {query_index} has no positive candidates")
        if values.min() < 0 or values.max() >= candidates.shape[0]:
            raise ValueError(f"query {query_index} contains an out-of-range positive index")
        normalized_positives.append(values)

    ranks = np.empty(queries.shape[0], dtype=np.int64)
    top1 = np.empty(queries.shape[0], dtype=np.int64)
    top1_scores = np.empty(queries.shape[0], dtype=np.float32)
    best_positive_indices = np.empty(queries.shape[0], dtype=np.int64)
    best_positive_scores = np.empty(queries.shape[0], dtype=np.float32)
    candidate_order = np.arange(candidates.shape[0], dtype=np.int64)
    for start in range(0, queries.shape[0], batch_size):
        stop = min(start + batch_size, queries.shape[0])
        scores = queries[start:stop] @ candidates.T
        block_top1 = np.argmax(scores, axis=1)
        top1[start:stop] = block_top1
        top1_scores[start:stop] = scores[np.arange(stop - start), block_top1]
        for local_index, positive in enumerate(normalized_positives[start:stop]):
            row = scores[local_index]
            positive_scores = row[positive]
            best_positive_score = float(np.max(positive_scores))
            tied_positive_indices = positive[positive_scores == best_positive_score]
            best_positive_index = int(np.min(tied_positive_indices))
            strictly_better = int(np.count_nonzero(row > best_positive_score))
            earlier_ties = int(
                np.count_nonzero(
                    (row == best_positive_score) & (candidate_order < best_positive_index)
                )
            )
            output_index = start + local_index
            ranks[output_index] = strictly_better + earlier_ties + 1
            best_positive_indices[output_index] = best_positive_index
            best_positive_scores[output_index] = best_positive_score
    return RankingResult(
        ranks=ranks,
        top1_indices=top1,
        top1_scores=top1_scores,
        best_positive_indices=best_positive_indices,
        best_positive_scores=best_positive_scores,
    )


def summarize_ranks(
    image_to_text: RankingResult,
    text_to_image: RankingResult,
    ks: Iterable[int] = (1, 5, 10),
) -> dict[str, float]:
    """Summarize two directional rank arrays with standard retrieval metrics."""
    cutoffs = _normalize_cutoffs(ks)
    metrics: dict[str, float] = {}
    recalls: list[float] = []
    for prefix, result in (("i2t", image_to_text), ("t2i", text_to_image)):
        ranks = np.asarray(result.ranks, dtype=np.int64)
        if ranks.ndim != 1 or ranks.size == 0 or np.any(ranks <= 0):
            raise ValueError(f"{prefix} ranks must be a non-empty positive one-dimensional array")
        for k in cutoffs:
            value = float(np.mean(ranks <= k) * 100.0)
            metrics[f"{prefix}_R@{k}"] = value
            recalls.append(value)
        metrics[f"{prefix}_MedR"] = float(np.median(ranks))
        metrics[f"{prefix}_MeanR"] = float(np.mean(ranks))
    metrics["mean_recall"] = float(np.mean(recalls))
    return metrics


def _validated_embeddings(value: np.ndarray, label: str) -> np.ndarray:
    embeddings = np.asarray(value, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError(f"{label} must be a non-empty two-dimensional array")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{label} must contain only finite values")
    return embeddings


def _normalize_cutoffs(ks: Iterable[int]) -> tuple[int, ...]:
    cutoffs = tuple(sorted(set(int(k) for k in ks)))
    if not cutoffs or cutoffs[0] <= 0:
        raise ValueError("all recall cutoffs must be positive integers")
    return cutoffs
