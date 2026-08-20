"""Mechanistic ablations and causal evidence utilities for ReCoAlign."""

from .causal import evaluate_token_interventions, intervene_structure_tokens
from .controls import FixedGraphEncoderControl, apply_supervision_ablation
from .failure import collect_failure_cases
from .parameters import parameter_audit, parameter_matched_control
from .probes import probe_structure_tokens, representation_comparison
from .registry import (
    ABLATION_SPECS,
    AblationSpec,
    load_mechanistic_registry,
    validate_mechanistic_registry,
)
from .reporting import generate_mechanistic_reports
from .runner import run_toy_mechanistic_suite
from .statistics import summarize_ablation_seeds
from .visualization import attention_summary, similarity_matrix, token_clusters

__all__ = [
    "ABLATION_SPECS",
    "AblationSpec",
    "FixedGraphEncoderControl",
    "apply_supervision_ablation",
    "attention_summary",
    "collect_failure_cases",
    "evaluate_token_interventions",
    "generate_mechanistic_reports",
    "intervene_structure_tokens",
    "load_mechanistic_registry",
    "parameter_audit",
    "parameter_matched_control",
    "probe_structure_tokens",
    "representation_comparison",
    "run_toy_mechanistic_suite",
    "similarity_matrix",
    "summarize_ablation_seeds",
    "token_clusters",
    "validate_mechanistic_registry",
]
