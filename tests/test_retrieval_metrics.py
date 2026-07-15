import numpy as np
import pytest

from recoalign.evaluation.retrieval import recall_at_k, summarize_bidirectional_retrieval


def test_recall_at_k_with_multiple_positives() -> None:
    similarity = np.array(
        [
            [0.9, 0.8, 0.1],
            [0.9, 0.7, 0.8],
        ]
    )
    positives = np.array(
        [
            [True, True, False],
            [False, False, True],
        ]
    )

    result = recall_at_k(similarity, positives, ks=(1, 2))

    assert result[1] == pytest.approx(50.0)
    assert result[2] == pytest.approx(100.0)


def test_bidirectional_summary_contains_mean_recall() -> None:
    similarity = np.eye(3)
    positives = np.eye(3, dtype=bool)

    result = summarize_bidirectional_retrieval(similarity, positives, ks=(1, 2))

    assert result["i2t_R@1"] == pytest.approx(100.0)
    assert result["t2i_R@1"] == pytest.approx(100.0)
    assert result["mean_recall"] == pytest.approx(100.0)


def test_recall_rejects_queries_without_positives() -> None:
    similarity = np.eye(2)
    positives = np.array([[True, False], [False, False]])

    with pytest.raises(ValueError, match="every query"):
        recall_at_k(similarity, positives)
