"""Preregistered mechanism classification that represents joint causality."""

from __future__ import annotations

from typing import Any

ALLOWED_OUTCOMES = {
    "SEMANTIC_PRIMARY_GO",
    "INTEGRATION_INDEPENDENT_GO",
    "JOINT_CAUSAL_GO",
    "NO-GO_ALTERNATIVE_MECHANISM",
    "INCONCLUSIVE",
}


def adjudicate_mechanism(
    analysis: dict[str, Any],
    config: dict[str, Any],
    *,
    integrity_passed: bool,
    power_passed: bool,
) -> dict[str, Any]:
    manipulation = analysis["manipulation_check"]
    estimands = analysis["estimands"]
    sesoi = float(config["statistics"]["superiority_sesoi"])
    required_hops = [str(value) for value in config["statistics"]["required_hop_depths"]]
    minimum_fraction = float(config["statistics"]["seed_replication_minimum_fraction"])

    global_gates = {
        "integrity": integrity_passed,
        "power": power_passed,
        "five_seeds": len(analysis["completed_seeds"]) == 5,
        "manipulation_check": bool(manipulation["passed"]),
    }
    if not all(global_gates.values()):
        return _decision(
            "INCONCLUSIVE",
            global_gates,
            reason="At least one preregistered global or manipulation gate failed.",
            evidence=[],
        )

    e1 = estimands["E1_semantic_rescue"]
    e2 = estimands["E2_relation_under_oracle"]
    e3 = estimands["E3_json_minus_triples"]
    e4 = estimands["E4_semantic_x_relation"]
    e5 = estimands["E5_semantic_x_format"]
    semantic_meaningful = _meaningful_positive(e1, sesoi) and _replicated(
        e1, minimum_fraction
    )
    relation_meaningful = (
        _meaningful_positive(e2, sesoi)
        and bool(e2["holm"]["reject_zero"])
        and _replicated(e2, minimum_fraction)
        and _hop_replicated(analysis, "E2_relation_under_oracle", required_hops, minimum_fraction)
    )
    format_meaningful = (
        _meaningful_absolute(e3, sesoi)
        and bool(e3["holm"]["reject_zero"])
        and _replicated_direction(e3, minimum_fraction)
        and _hop_replicated(analysis, "E3_json_minus_triples", required_hops, minimum_fraction)
    )
    relation_equivalent = bool(e2["tost"]["equivalent"])
    format_equivalent = bool(e3["tost"]["equivalent"])
    interaction_relation = _stable_interaction(e4, minimum_fraction)
    interaction_format = _stable_interaction(e5, minimum_fraction)
    residual_meaningful = relation_meaningful or format_meaningful
    interaction_stable = interaction_relation or interaction_format
    evidence = [
        f"E1 semantic rescue={e1['mean']:.4f}; meaningful={semantic_meaningful}.",
        f"E2 oracle-controlled relation={e2['mean']:.4f}; meaningful={relation_meaningful}; "
        f"equivalent={relation_equivalent}.",
        f"E3 oracle-controlled JSON-triples={e3['mean']:.4f}; meaningful={format_meaningful}; "
        f"equivalent={format_equivalent}.",
        f"Stable interaction: relation={interaction_relation}, format={interaction_format}.",
    ]

    if semantic_meaningful and residual_meaningful and interaction_stable:
        return _decision(
            "JOINT_CAUSAL_GO",
            global_gates,
            reason=(
                "Semantic rescue is meaningful and a controlled integration effect plus its "
                "semantic interaction replicate under oracle semantics."
            ),
            evidence=evidence,
            independent_replication=True,
            construct_validity=True,
        )
    if semantic_meaningful and relation_equivalent and format_equivalent:
        return _decision(
            "SEMANTIC_PRIMARY_GO",
            global_gates,
            reason=(
                "Semantic rescue is meaningful and both oracle-controlled relation and format "
                "effects satisfy preregistered equivalence."
            ),
            evidence=evidence,
            independent_replication=True,
        )
    if residual_meaningful and not interaction_stable:
        return _decision(
            "INTEGRATION_INDEPENDENT_GO",
            global_gates,
            reason=(
                "A controlled relation or format effect remains meaningful under sufficient "
                "oracle semantics without a reproducible semantic interaction."
            ),
            evidence=evidence,
            independent_replication=True,
        )
    return _decision(
        "NO-GO_ALTERNATIVE_MECHANISM",
        global_gates,
        reason="All gates passed, but the effect pattern matches no preregistered mechanism class.",
        evidence=evidence,
    )


def _meaningful_positive(summary: dict[str, Any], sesoi: float) -> bool:
    return bool(
        summary["mean"] >= sesoi and summary["confidence_interval"]["lower"] > 0.0
    )


def _meaningful_absolute(summary: dict[str, Any], sesoi: float) -> bool:
    interval = summary["confidence_interval"]
    return bool(
        abs(summary["mean"]) >= sesoi
        and (interval["lower"] > 0.0 or interval["upper"] < 0.0)
    )


def _replicated(summary: dict[str, Any], fraction: float) -> bool:
    return bool(summary["positive_seed_fraction"] >= fraction)


def _replicated_direction(summary: dict[str, Any], fraction: float) -> bool:
    return bool(
        max(summary["positive_seed_fraction"], summary["negative_seed_fraction"]) >= fraction
    )


def _stable_interaction(summary: dict[str, Any], fraction: float) -> bool:
    interval = summary["confidence_interval"]
    return bool(
        (interval["lower"] > 0.0 or interval["upper"] < 0.0)
        and _replicated_direction(summary, fraction)
    )


def _hop_replicated(
    analysis: dict[str, Any], estimand: str, hops: list[str], fraction: float
) -> bool:
    for hop in hops:
        summary = analysis["hop_depth"][hop][estimand]
        if not _replicated_direction(summary, fraction):
            return False
    return True


def _decision(
    outcome: str,
    gates: dict[str, bool],
    *,
    reason: str,
    evidence: list[str],
    independent_replication: bool = False,
    construct_validity: bool = False,
) -> dict[str, Any]:
    if outcome not in ALLOWED_OUTCOMES:
        raise ValueError(f"unregistered causal-separation outcome: {outcome}")
    return {
        "outcome": outcome,
        "reason": reason,
        "evidence": evidence,
        "global_gates": gates,
        "authorization": {
            "independent_replication_allowed": independent_replication,
            "construct_validity_analysis_allowed": construct_validity,
            "model_development_allowed": False,
            "paper_writing_allowed": False,
        },
    }


__all__ = ["ALLOWED_OUTCOMES", "adjudicate_mechanism"]
