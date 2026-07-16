from __future__ import annotations

import numpy as np

from recoalign.benchmarks.caption_multisets import WINOGROUND_CONTENT_MULTISET
from recoalign.evaluation.diagnostics import (
    evaluate_multichoice_scores,
    evaluate_paired_matrix_scores,
    summarize_multichoice,
    summarize_paired_matrix,
    token_multiset_match_rate,
)


def test_multichoice_is_strict_and_reports_blind_controls() -> None:
    result = evaluate_multichoice_scores(
        [np.asarray([0.9, 0.1]), np.asarray([0.2, 0.2, 0.1])],
        [0, 1],
    )
    metrics = summarize_multichoice(
        result,
        ["relation", "attribution"],
        [["short", "a longer caption"], ["one", "two words", "three word caption"]],
        [0, 1],
    )

    assert metrics["accuracy"] == 50.0
    assert metrics["tie_rate"] == 50.0
    assert metrics["accuracy/relation"] == 100.0
    assert metrics["accuracy/attribution"] == 0.0
    assert "blind/majority_index_accuracy" in metrics
    assert "blind/shortest_caption_accuracy" in metrics
    assert "blind/longest_caption_accuracy" in metrics


def test_paired_matrix_reports_both_directions_and_group() -> None:
    result = evaluate_paired_matrix_scores(
        np.asarray(
            [
                [[0.9, 0.1], [0.2, 0.8]],
                [[0.5, 0.5], [0.1, 0.7]],
            ]
        )
    )
    metrics = summarize_paired_matrix(
        result,
        ["clean", "tie"],
        tags=[["spatial"], ["spatial", "unusual"]],
    )

    assert metrics["image_to_text_accuracy"] == 50.0
    assert metrics["text_to_image_accuracy"] == 100.0
    assert metrics["group_accuracy"] == 50.0
    assert metrics["tie_rate"] == 50.0
    assert metrics["group_accuracy/tag/spatial"] == 50.0
    assert metrics["group_accuracy/tag/unusual"] == 0.0


def test_token_multiset_match_rate_is_order_invariant() -> None:
    assert token_multiset_match_rate([("a red cup", "cup red a")]) == 100.0
    assert token_multiset_match_rate([("a red cup", "cup red a"), ("one", "two")]) == 50.0


def test_token_multiset_match_rate_supports_official_morpheme_cases() -> None:
    pairs = [
        ("a caterpillar with some plants", "a plant with some caterpillars"),
        ("first the cream, then the jam", "first the jam, then the cream"),
        ("The dog rides without a visible tongue", "The dog rides with a visible tongue out"),
    ]

    assert (
        token_multiset_match_rate(pairs, method=WINOGROUND_CONTENT_MULTISET) == 100.0
    )
