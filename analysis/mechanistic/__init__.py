"""Compatibility exports for the canonical mechanistic analysis package."""

from recoalign.analysis.mechanistic import (
    ABLATION_SPECS,
    AblationSpec,
    FixedGraphEncoderControl,
    apply_supervision_ablation,
    attention_summary,
    collect_failure_cases,
    evaluate_token_interventions,
    intervene_structure_tokens,
    load_mechanistic_registry,
    parameter_audit,
    parameter_matched_control,
    probe_structure_tokens,
    representation_comparison,
    run_toy_mechanistic_suite,
    similarity_matrix,
    summarize_ablation_seeds,
    token_clusters,
    validate_mechanistic_registry,
)

__all__ = [
    "ABLATION_SPECS", "AblationSpec", "FixedGraphEncoderControl", "apply_supervision_ablation",
    "attention_summary", "collect_failure_cases", "evaluate_token_interventions",
    "intervene_structure_tokens", "load_mechanistic_registry", "parameter_audit",
    "parameter_matched_control", "probe_structure_tokens", "representation_comparison",
    "run_toy_mechanistic_suite", "similarity_matrix", "summarize_ablation_seeds",
    "token_clusters", "validate_mechanistic_registry",
]
