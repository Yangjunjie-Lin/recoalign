"""Bias-aware metrics for compositional vision-language benchmarks."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from recoalign.benchmarks.caption_multisets import (
    WHITESPACE_TOKEN_MULTISET,
    caption_multiset_matches,
)


@dataclass(frozen=True)
class MultiChoiceEvaluation:
    """Tie-aware results for one-image, many-caption compositional choices."""

    predicted_indices: np.ndarray
    correct_scores: np.ndarray
    top_scores: np.ndarray
    margins: np.ndarray
    correct: np.ndarray
    ties: np.ndarray


@dataclass(frozen=True)
class PairedMatrixEvaluation:
    """Results for a two-image by two-caption matching problem."""

    scores: np.ndarray
    image_to_text_correct: np.ndarray
    text_to_image_correct: np.ndarray
    group_correct: np.ndarray
    ties: np.ndarray
    image_to_text_margins: np.ndarray
    text_to_image_margins: np.ndarray


def evaluate_multichoice_scores(
    score_rows: Sequence[np.ndarray],
    correct_indices: Sequence[int],
) -> MultiChoiceEvaluation:
    """Evaluate variable-width caption choices with strict, tie-aware correctness."""
    if not score_rows:
        raise ValueError("score_rows must be non-empty")
    if len(score_rows) != len(correct_indices):
        raise ValueError("score_rows and correct_indices must align")

    predicted_indices: list[int] = []
    correct_scores: list[float] = []
    top_scores: list[float] = []
    margins: list[float] = []
    correct: list[bool] = []
    ties: list[bool] = []

    for row_index, (raw_scores, correct_index) in enumerate(
        zip(score_rows, correct_indices, strict=True)
    ):
        scores = np.asarray(raw_scores, dtype=np.float64)
        if scores.ndim != 1 or scores.size < 2 or not np.isfinite(scores).all():
            raise ValueError(
                f"score row {row_index} must be a finite one-dimensional array "
                "with at least two entries"
            )
        if (
            not isinstance(correct_index, int)
            or isinstance(correct_index, bool)
            or not 0 <= correct_index < scores.size
        ):
            raise ValueError(f"correct index {row_index} is invalid")

        predicted_index = int(np.argmax(scores))
        correct_score = float(scores[correct_index])
        best_other_score = float(np.delete(scores, correct_index).max())
        top_score = float(scores.max())

        predicted_indices.append(predicted_index)
        correct_scores.append(correct_score)
        top_scores.append(top_score)
        margins.append(correct_score - best_other_score)
        correct.append(correct_score > best_other_score)
        ties.append(bool(np.count_nonzero(scores == top_score) > 1))

    return MultiChoiceEvaluation(
        predicted_indices=np.asarray(predicted_indices, dtype=np.int64),
        correct_scores=np.asarray(correct_scores, dtype=np.float64),
        top_scores=np.asarray(top_scores, dtype=np.float64),
        margins=np.asarray(margins, dtype=np.float64),
        correct=np.asarray(correct, dtype=bool),
        ties=np.asarray(ties, dtype=bool),
    )


def summarize_multichoice(
    result: MultiChoiceEvaluation,
    subsets: Sequence[str],
    caption_lists: Sequence[Sequence[str]],
    correct_indices: Sequence[int],
) -> dict[str, float]:
    """Summarize ARO-style results and simple blind dataset heuristics."""
    size = len(result.correct)
    if not (size == len(subsets) == len(caption_lists) == len(correct_indices)):
        raise ValueError("multi-choice summary inputs must align")

    metrics = {
        "accuracy": float(result.correct.mean() * 100.0),
        "tie_rate": float(result.ties.mean() * 100.0),
        "mean_margin": float(result.margins.mean()),
    }

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, subset in enumerate(subsets):
        grouped[subset].append(index)

    subset_accuracies: list[float] = []
    for subset, indices in sorted(grouped.items()):
        accuracy = float(result.correct[indices].mean() * 100.0)
        metrics[f"accuracy/{subset}"] = accuracy
        subset_accuracies.append(accuracy)
    metrics["macro_subset_accuracy"] = float(np.mean(subset_accuracies))

    correct_index_counts = Counter(correct_indices)
    majority_index, _ = correct_index_counts.most_common(1)[0]
    metrics["blind/majority_index_accuracy"] = float(
        np.mean(np.asarray(correct_indices) == majority_index) * 100.0
    )

    shortest_hits: list[bool] = []
    longest_hits: list[bool] = []
    for captions, correct_index in zip(caption_lists, correct_indices, strict=True):
        lengths = np.asarray([len(caption.split()) for caption in captions])
        shortest_hits.append(correct_index == int(np.argmin(lengths)))
        longest_hits.append(correct_index == int(np.argmax(lengths)))
    metrics["blind/shortest_caption_accuracy"] = float(np.mean(shortest_hits) * 100.0)
    metrics["blind/longest_caption_accuracy"] = float(np.mean(longest_hits) * 100.0)
    return metrics


def evaluate_paired_matrix_scores(scores: np.ndarray) -> PairedMatrixEvaluation:
    """Evaluate a batch of two-image by two-caption score matrices.

    The score convention is ``scores[sample, image_index, caption_index]``. Correct pairs are
    diagonal. Every comparison uses strict greater-than; ties are incorrect and reported.
    """
    matrix = np.asarray(scores, dtype=np.float64)
    if (
        matrix.ndim != 3
        or matrix.shape[1:] != (2, 2)
        or matrix.shape[0] == 0
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("scores must have shape [N, 2, 2] with finite values")

    score_00 = matrix[:, 0, 0]
    score_01 = matrix[:, 0, 1]
    score_10 = matrix[:, 1, 0]
    score_11 = matrix[:, 1, 1]

    image_to_text_margins = np.minimum(score_00 - score_01, score_11 - score_10)
    text_to_image_margins = np.minimum(score_00 - score_10, score_11 - score_01)
    image_to_text_correct = image_to_text_margins > 0
    text_to_image_correct = text_to_image_margins > 0
    group_correct = image_to_text_correct & text_to_image_correct
    ties = (
        (score_00 == score_01)
        | (score_11 == score_10)
        | (score_00 == score_10)
        | (score_11 == score_01)
    )

    return PairedMatrixEvaluation(
        scores=matrix,
        image_to_text_correct=image_to_text_correct,
        text_to_image_correct=text_to_image_correct,
        group_correct=group_correct,
        ties=ties,
        image_to_text_margins=image_to_text_margins,
        text_to_image_margins=text_to_image_margins,
    )


def summarize_paired_matrix(
    result: PairedMatrixEvaluation,
    categories: Sequence[str],
    tags: Sequence[Sequence[str]] | None = None,
) -> dict[str, float]:
    """Return overall, category, and optional tag metrics for Winoground/BiVLC."""
    if len(result.group_correct) != len(categories):
        raise ValueError("categories must align with paired-matrix results")

    metrics = {
        "image_to_text_accuracy": float(result.image_to_text_correct.mean() * 100.0),
        "text_to_image_accuracy": float(result.text_to_image_correct.mean() * 100.0),
        "group_accuracy": float(result.group_correct.mean() * 100.0),
        "tie_rate": float(result.ties.mean() * 100.0),
        "mean_image_to_text_margin": float(result.image_to_text_margins.mean()),
        "mean_text_to_image_margin": float(result.text_to_image_margins.mean()),
    }

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, category in enumerate(categories):
        grouped[category].append(index)

    group_accuracies: list[float] = []
    for category, indices in sorted(grouped.items()):
        image_to_text = float(result.image_to_text_correct[indices].mean() * 100.0)
        text_to_image = float(result.text_to_image_correct[indices].mean() * 100.0)
        group = float(result.group_correct[indices].mean() * 100.0)
        metrics[f"image_to_text_accuracy/{category}"] = image_to_text
        metrics[f"text_to_image_accuracy/{category}"] = text_to_image
        metrics[f"group_accuracy/{category}"] = group
        group_accuracies.append(group)
    metrics["macro_category_group_accuracy"] = float(np.mean(group_accuracies))

    if tags is not None:
        if len(tags) != len(categories):
            raise ValueError("tags must align with paired-matrix results")
        tag_groups: dict[str, list[int]] = defaultdict(list)
        for index, sample_tags in enumerate(tags):
            for tag in sample_tags:
                tag_groups[tag].append(index)
        for tag, indices in sorted(tag_groups.items()):
            metrics[f"group_accuracy/tag/{tag}"] = float(
                result.group_correct[indices].mean() * 100.0
            )
    return metrics


def token_multiset_match_rate(
    caption_pairs: Sequence[tuple[str, str]],
    *,
    method: str = WHITESPACE_TOKEN_MULTISET,
) -> float:
    """Return the match rate under an explicit caption-content multiset policy."""
    if not caption_pairs:
        raise ValueError("caption_pairs must be non-empty")

    return float(
        np.mean(
            [
                caption_multiset_matches(first, second, method=method)
                for first, second in caption_pairs
            ]
        )
        * 100.0
    )
