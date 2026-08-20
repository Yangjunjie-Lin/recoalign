"""Mechanism, capability-preservation, token, and failure analyses."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def capability_preservation(
    original: list[dict[str, Any]],
    recoalign: list[dict[str, Any]],
    *,
    maximum_degradation: float = 0.02,
) -> dict[str, Any]:
    """Compare paired general-capability predictions without hiding regressions."""

    paired = _paired_correct(original, recoalign)
    if not paired:
        raise ValueError("capability preservation requires paired predictions")
    baseline = sum(first for first, _second in paired) / len(paired)
    method = sum(second for _first, second in paired) / len(paired)
    degradation = baseline - method
    return {
        "original_accuracy": baseline,
        "recoalign_accuracy": method,
        "degradation": degradation,
        "maximum_allowed_degradation": maximum_degradation,
        "preserved": degradation <= maximum_degradation,
        "n_pairs": len(paired),
    }


def mechanism_consistency(
    diagnosis: dict[str, Any],
    method_comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    """Connect SAS/StAS/RES diagnosis to measured method gains."""

    sas = _first_number(
        diagnosis,
        (("diagnosis", "sas", "estimate"), ("scores", "sas", "mean")),
    )
    stas = _first_number(
        diagnosis,
        (("diagnosis", "stas", "estimate"), ("scores", "stas", "mean")),
    )
    oracle_gain = _first_number(
        diagnosis,
        (
            ("diagnosis", "oracle_graph_gain", "estimate"),
            ("scores", "oracle_graph_gain", "mean"),
        ),
    )
    gains = [
        float(row["gain"])
        for row in method_comparisons
        if isinstance(row.get("gain"), (int, float)) and not isinstance(row.get("gain"), bool)
    ]
    eligible_diagnosis = diagnosis.get("claim_status") == "scientific_evidence"
    diagnosis_pattern = eligible_diagnosis and sas is not None and stas is not None and sas > stas
    positive_method_pattern = bool(gains) and sum(gain > 0 for gain in gains) / len(gains) >= 0.8
    return {
        "semantic_availability": sas,
        "structured_accessibility": stas,
        "oracle_graph_gain": oracle_gain,
        "diagnosed_interface_gap": diagnosis_pattern,
        "method_gain_values": gains,
        "positive_method_gain_fraction": (
            sum(gain > 0 for gain in gains) / len(gains) if gains else None
        ),
        "mechanism_consistent": diagnosis_pattern and positive_method_pattern,
        "scientific_status": (
            "pending_real_diagnosis"
            if not eligible_diagnosis
            else "assessed"
            if gains
            else "pending_method_results"
        ),
    }


def structure_token_analysis(
    structure_tokens: np.ndarray,
    labels: list[str] | np.ndarray,
) -> dict[str, Any]:
    """Report within-label and between-label cosine similarity for learned tokens."""

    values = np.asarray(structure_tokens, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2:
        raise ValueError("structure tokens must have shape [samples, tokens, dimension]")
    label_values = [str(value) for value in labels]
    if len(label_values) != values.shape[0]:
        raise ValueError("structure token labels must match sample count")
    pooled = values.mean(axis=1)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    normalized = pooled / np.maximum(norms, 1e-12)
    similarities = normalized @ normalized.T
    within: list[float] = []
    between: list[float] = []
    for first in range(len(label_values)):
        for second in range(first + 1, len(label_values)):
            target = within if label_values[first] == label_values[second] else between
            target.append(float(similarities[first, second]))
    return {
        "samples": values.shape[0],
        "num_tokens": values.shape[1],
        "dimension": values.shape[2],
        "within_label_similarity": sum(within) / len(within) if within else None,
        "between_label_similarity": sum(between) / len(between) if between else None,
        "separation": (
            sum(within) / len(within) - sum(between) / len(between)
            if within and between
            else None
        ),
    }


def compare_failure_taxonomy(
    original: list[dict[str, Any]], recoalign: list[dict[str, Any]]
) -> dict[str, Any]:
    """Count paired before/after failures and reductions by causal category."""

    original_lookup = {str(row["sample_id"]): row for row in original}
    method_lookup = {str(row["sample_id"]): row for row in recoalign}
    common = sorted(set(original_lookup) & set(method_lookup))
    if not common:
        raise ValueError("failure comparison requires paired sample IDs")
    before: Counter[str] = Counter()
    after: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    for sample_id in common:
        left = original_lookup[sample_id]
        right = method_lookup[sample_id]
        failure = str(left.get("failure_type") or "correct")
        after_failure = str(right.get("failure_type") or "correct")
        if not bool(left.get("correct")):
            before[failure] += 1
        if not bool(right.get("correct")):
            after[after_failure] += 1
        transitions[f"{failure}->{after_failure}"] += 1
    categories = sorted(set(before) | set(after))
    return {
        "n_pairs": len(common),
        "before": dict(before),
        "after": dict(after),
        "reduction": {category: before[category] - after[category] for category in categories},
        "transitions": dict(transitions),
    }


def _paired_correct(
    original: list[dict[str, Any]], recoalign: list[dict[str, Any]]
) -> list[tuple[float, float]]:
    left = {str(row["sample_id"]): float(bool(row["correct"])) for row in original}
    right = {str(row["sample_id"]): float(bool(row["correct"])) for row in recoalign}
    return [(left[key], right[key]) for key in sorted(set(left) & set(right))]


def _nested_number(payload: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _first_number(
    payload: dict[str, Any], paths: tuple[tuple[str, ...], ...]
) -> float | None:
    for path in paths:
        value = _nested_number(payload, path)
        if value is not None:
            return value
    return None


__all__ = [
    "capability_preservation",
    "compare_failure_taxonomy",
    "mechanism_consistency",
    "structure_token_analysis",
]
