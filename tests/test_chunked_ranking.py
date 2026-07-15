import numpy as np

from recoalign.evaluation.retrieval import rank_queries, summarize_ranks


def test_chunked_ranking_matches_expected_recall() -> None:
    images = np.eye(3, dtype=np.float32)
    texts = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 0.9],
        ],
        dtype=np.float32,
    )
    i2t = rank_queries(images, texts, [[0, 1], [2, 3], [4, 5]], batch_size=1)
    t2i = rank_queries(texts, images, [[0], [0], [1], [1], [2], [2]], batch_size=2)
    metrics = summarize_ranks(i2t, t2i, (1, 5, 10))
    assert metrics["i2t_R@1"] == 100.0
    assert metrics["t2i_R@1"] == 100.0
    assert metrics["mean_recall"] == 100.0


def test_chunked_ranking_uses_stable_candidate_index_for_ties() -> None:
    queries = np.asarray([[1.0, 0.0]], dtype=np.float32)
    candidates = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    result = rank_queries(queries, candidates, [[1]], batch_size=1)
    assert result.ranks.tolist() == [2]
    assert result.top1_indices.tolist() == [0]
    assert result.best_positive_indices.tolist() == [1]
