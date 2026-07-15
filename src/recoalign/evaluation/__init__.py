"""Evaluation utilities for retrieval and compositional benchmarks."""

from recoalign.evaluation.baseline import BaselineEvaluation, evaluate_baseline
from recoalign.evaluation.compositional import evaluate_pairwise_scores, summarize_pairwise
from recoalign.evaluation.diagnostics import (
    evaluate_multichoice_scores,
    evaluate_paired_matrix_scores,
    summarize_multichoice,
    summarize_paired_matrix,
    token_multiset_match_rate,
)
from recoalign.evaluation.retrieval import (
    rank_queries,
    recall_at_k,
    summarize_bidirectional_retrieval,
    summarize_ranks,
)

__all__ = [
    "BaselineEvaluation",
    "evaluate_baseline",
    "evaluate_multichoice_scores",
    "evaluate_paired_matrix_scores",
    "evaluate_pairwise_scores",
    "rank_queries",
    "recall_at_k",
    "summarize_bidirectional_retrieval",
    "summarize_multichoice",
    "summarize_paired_matrix",
    "summarize_pairwise",
    "summarize_ranks",
    "token_multiset_match_rate",
]
