from __future__ import annotations

from typing import Any

from .config import ExperimentConfig
from .probes import SEMANTICS


def _accuracy(probe_results: dict[str, Any], stage: str, semantic: str) -> float:
    for row in probe_results["rows"]:
        if row["stage"] == stage and row["semantic"] == semantic:
            return float(row["accuracy"])
    raise KeyError((stage, semantic))


def determine_decision(config: ExperimentConfig, probe_results: dict[str, Any]) -> dict[str, Any]:
    za_gates = {
        f"za_{semantic}_accuracy_gte_{int(config.min_za_accuracy * 100)}pct": (
            _accuracy(probe_results, "Za", semantic) >= config.min_za_accuracy
        )
        for semantic in SEMANTICS
    }
    identity_gate = probe_results["identity"]["max_abs_difference"] < 0.02
    layer_audits: dict[str, Any] = {}
    passing_layers: list[int] = []
    for layer in config.layer_indices[1:]:
        stage = f"L{layer:02d}_visual"
        drop = probe_results["drops"][stage]
        selective = probe_results["selectivity"][stage]
        gates = {
            "object_drop_lt_10pp": drop["object"]["drop"] < config.max_object_drop,
            "relation_drop_gt_15pp": drop["relation"]["drop"] > config.min_specific_drop,
            "composition_drop_gt_15pp": (drop["composition"]["drop"] > config.min_specific_drop),
            "relation_drop_ci_above_zero": drop["relation"]["ci95_low"] > 0,
            "composition_drop_ci_above_zero": drop["composition"]["ci95_low"] > 0,
            "relation_minus_object_ci_above_zero": (selective["relation"]["ci95_low"] > 0),
            "composition_minus_object_ci_above_zero": (selective["composition"]["ci95_low"] > 0),
        }
        passed = all(gates.values())
        layer_audits[stage] = {"gates": gates, "passes_specific_degradation": passed}
        if passed:
            passing_layers.append(layer)

    protocol_gates = {
        "full_protocol": config.scientific_protocol_valid(),
        "za_layer0_identity": identity_gate,
        "all_linear_probes_converged": probe_results["all_probes_converged"],
        **za_gates,
    }
    go = bool(passing_layers) and all(protocol_gates.values())
    max_rel = max(
        probe_results["drops"][f"L{layer:02d}_visual"]["relation"]["drop"]
        for layer in config.layer_indices[1:]
    )
    max_comp = max(
        probe_results["drops"][f"L{layer:02d}_visual"]["composition"]["drop"]
        for layer in config.layer_indices[1:]
    )
    max_obj = max(
        probe_results["drops"][f"L{layer:02d}_visual"]["object"]["drop"]
        for layer in config.layer_indices[1:]
    )
    if go:
        pattern = "specific relation/composition degradation inside frozen LLM"
    elif max(max_rel, max_comp) <= config.min_specific_drop:
        pattern = "no material semantic degradation inside the tested LLM layers"
    elif max_obj >= config.max_object_drop:
        pattern = "generic or non-specific hidden-state degradation"
    else:
        pattern = "partial/inconclusive degradation; pre-registered selectivity gates failed"
    return {
        "decision": "GO" if go else "NO-GO",
        "semantic_utilization_failure": bool(go),
        "pattern": pattern,
        "passing_layers": passing_layers,
        "protocol_gates": protocol_gates,
        "layer_audits": layer_audits,
        "observed_max_drops": {
            "object": max_obj,
            "relation": max_rel,
            "composition": max_comp,
        },
        "next_phase": "Causal Intervention design" if go else None,
    }
