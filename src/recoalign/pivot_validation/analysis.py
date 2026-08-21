"""Statistics and evidence-first mechanism classification for PIVOT_EXP_A."""

from __future__ import annotations

import itertools
import statistics
from collections import defaultdict
from typing import Any

from evaluation.statistics import multiple_seed_summary, paired_bootstrap_test

from .design import A1_TASKS, A2_CONDITIONS, A3_CONDITIONS


def analyze_predictions(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    if not rows:
        raise ValueError("pivot analysis requires predictions")
    seeds = sorted({int(row["seed"]) for row in rows})
    metrics = {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A",
        "completed_seeds": seeds,
        "seed_count": len(seeds),
        "prediction_count": len(rows),
        "A1": _analyze_a1(rows, seeds, config),
        "A2": _analyze_a2(rows, seeds, config),
        "A3": _analyze_a3(rows, seeds, config),
    }
    metrics["mechanism"] = classify_mechanism(metrics, config)
    return metrics


def classify_mechanism(metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    a1 = metrics["A1"]
    statuses = {task: a1["tasks"][task]["availability"] for task in A1_TASKS}
    object_high = statuses["object_shape"] == "HIGH"
    attribute_high = statuses["attribute_color"] == "HIGH"
    relation_high = statuses["direct_relation"] == "HIGH"
    object_low = statuses["object_shape"] == "LOW"
    attribute_low = statuses["attribute_color"] == "LOW"
    relation_low = statuses["direct_relation"] == "LOW"
    indeterminate = any(value == "INDETERMINATE" for value in statuses.values())
    stable = metrics["A2"]["stable_effects"]
    complete_stable = bool(stable["complete_contribution"])
    relation_stable = bool(stable["relation_contribution"])
    format_status = metrics["A3"]["format_classification"]
    image_accuracy = float(metrics["A2"]["conditions"]["image_only"]["summary"]["mean"])
    complete_accuracy = float(
        metrics["A2"]["conditions"]["complete_evidence"]["summary"]["mean"]
    )

    classification = "MULTIPLE_EXPLANATIONS_REMAIN"
    evidence: list[str] = []
    direction = "Retain PH001 as unresolved; do not select a mechanism."
    reproducible = False

    if not indeterminate and (object_low or attribute_low) and relation_low:
        classification = "PRIMITIVE_SEMANTIC_REPRESENTATION_LIMITATION"
        evidence = [
            "At least one object/attribute dimension and direct relation are LOW under A1.",
            f"A2 complete-evidence contribution stable={complete_stable}.",
        ]
        direction = (
            "Prioritize visual primitive/grounding construct validation; relational integration is "
            "not identifiable until primitive availability improves."
        )
        reproducible = True
    elif not indeterminate and object_high and attribute_high and relation_low and complete_stable:
        classification = "RELATIONAL_GROUNDING_FAILURE"
        evidence = [
            "Object shape and color are HIGH while direct relation is LOW under A1.",
            "Complete external evidence has a stable positive contribution under A2.",
            f"Relation-only contribution stable={relation_stable}; A3={format_status}.",
        ]
        direction = (
            "Study relation-selective visual grounding before any reasoning adapter; treat format "
            "sensitivity as a secondary moderator."
        )
        reproducible = True
    elif not indeterminate and object_high and attribute_high and relation_high:
        if format_status == "FORMAT_SENSITIVE":
            classification = "FORMAT_CONDITIONED_RELATIONAL_INTEGRATION_FAILURE"
            evidence = [
                "All A1 primitive dimensions are HIGH.",
                "A3 contains a stable token-matched format effect.",
                f"A2 complete-evidence contribution stable={complete_stable}.",
            ]
            direction = (
                "Investigate representation-format consumption with frozen models; do not infer a "
                "missing graph interface or design an adapter."
            )
            reproducible = True
        elif (
            format_status == "FORMAT_INVARIANT"
            and complete_stable
            and image_accuracy < config["A1"]["high_availability"]["minimum_mean_accuracy"]
            and complete_accuracy >= config["A1"]["high_availability"]["minimum_mean_accuracy"]
        ):
            classification = "COMPOSITIONAL_REASONING_BOTTLENECK"
            evidence = [
                "All primitive dimensions are HIGH.",
                "A3 formats are equivalent within the registered margin.",
                "Complete evidence closes a low image-only compositional score.",
            ]
            direction = (
                "Study composition and multi-hop execution with frozen models; primitive semantics "
                "and representation format are not the primary bottleneck."
            )
            reproducible = True
    elif not indeterminate and (object_low or attribute_low) and relation_high:
        classification = "OBJECT_ATTRIBUTE_SEMANTIC_LIMITATION"
        evidence = [
            "At least one object/attribute dimension is LOW while direct relation is HIGH.",
            f"A2 complete-evidence contribution stable={complete_stable}.",
        ]
        direction = "Audit object/attribute rendering and behavioral construct validity."
        reproducible = True

    final_seed_count = len(metrics["completed_seeds"])
    integrity = bool(metrics.get("integrity_passed", True))
    if final_seed_count < len(config["study"]["seeds"]) or not integrity:
        decision = "INCONCLUSIVE"
        reason = "Five complete seeds and all integrity gates are required for a final decision."
    elif classification == "MULTIPLE_EXPLANATIONS_REMAIN" or not reproducible:
        decision = "NO-GO"
        reason = "The registered tests do not distinguish one reproducible primary mechanism."
    else:
        decision = "GO"
        reason = (
            "One preregistered primary mechanism classification is reproducible "
            "across five seeds."
        )
    return {
        "classification": classification,
        "evidence": evidence,
        "updated_research_direction": direction,
        "reproducible": reproducible,
        "decision": decision,
        "reason": reason,
        "hypothesis_status": {
            "H-A": _hypothesis_a_status(statuses),
            "H-B": _hypothesis_b_status(format_status, stable, classification),
        },
        "paper_writing_allowed": False,
        "model_development_allowed": False,
    }


def _analyze_a1(
    rows: list[dict[str, Any]], seeds: list[int], config: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "chance_accuracy": float(config["A1"]["chance_accuracy"]),
        "tasks": {},
    }
    semantic_by_seed: dict[int, list[float]] = defaultdict(list)
    for task in A1_TASKS:
        values = [_accuracy(rows, seed=seed, experiment="A1", task=task) for seed in seeds]
        summary = _summary(values, config, offset=len(result["tasks"]))
        chance = result["chance_accuracy"]
        normalized = [(value - chance) / (1.0 - chance) for value in values]
        for seed, value in zip(seeds, values, strict=True):
            semantic_by_seed[seed].append(value)
        result["tasks"][task] = {
            "summary": summary,
            "chance_normalized": _summary(normalized, config, offset=20 + len(result["tasks"])),
            "availability": _availability(summary, config),
            "n_per_seed": [
                _count(rows, seed=seed, experiment="A1", task=task) for seed in seeds
            ],
        }
    semantic_values = [statistics.fmean(semantic_by_seed[seed]) for seed in seeds]
    result["semantic_score"] = _summary(semantic_values, config, offset=40)
    result["all_high"] = all(
        row["availability"] == "HIGH" for row in result["tasks"].values()
    )
    return result


def _analyze_a2(
    rows: list[dict[str, Any]], seeds: list[int], config: dict[str, Any]
) -> dict[str, Any]:
    conditions = {
        condition: {
            "summary": _summary(
                [
                    _accuracy(rows, seed=seed, experiment="A2", condition=condition)
                    for seed in seeds
                ],
                config,
                offset=index + 100,
            ),
            "by_question_type": _slice_accuracies(rows, "A2", condition, "task"),
            "by_hop_depth": _slice_accuracies(rows, "A2", condition, "hop_depth"),
        }
        for index, condition in enumerate(A2_CONDITIONS)
    }
    formulas = {
        "object_contribution": ("object_evidence", "image_only"),
        "relation_contribution": ("relation_evidence", "image_only"),
        "complete_contribution": ("complete_evidence", "image_only"),
        "integration_surplus": ("complete_evidence", "max_component"),
        "additive_synergy": ("complete_evidence", "additive_components"),
    }
    effects: dict[str, Any] = {}
    for index, (name, formula) in enumerate(formulas.items()):
        seed_values = [
            _a2_seed_effect(rows, seed=seed, formula=formula) for seed in seeds
        ]
        scene_left, scene_right = _a2_scene_pairs(rows, formula=formula)
        effects[name] = {
            "summary": _summary(seed_values, config, offset=200 + index),
            "paired_scene_test": paired_bootstrap_test(
                scene_left,
                scene_right,
                alternative="greater",
                samples=int(config["statistics"]["bootstrap_samples"]),
                seed=int(config["statistics"]["statistics_seed"]) + 300 + index,
            ),
        }
    stable = {
        name: _stable_positive(effect["summary"], config["A2"]["stable_effect"])
        for name, effect in effects.items()
    }
    return {"conditions": conditions, "effects": effects, "stable_effects": stable}


def _analyze_a3(
    rows: list[dict[str, Any]], seeds: list[int], config: dict[str, Any]
) -> dict[str, Any]:
    conditions = {
        condition: {
            "summary": _summary(
                [
                    _accuracy(rows, seed=seed, experiment="A3", condition=condition)
                    for seed in seeds
                ],
                config,
                offset=index + 400,
            ),
            "by_question_type": _slice_accuracies(rows, "A3", condition, "task"),
            "by_hop_depth": _slice_accuracies(rows, "A3", condition, "hop_depth"),
        }
        for index, condition in enumerate(A3_CONDITIONS)
    }
    pairwise: dict[str, Any] = {}
    sensitive = False
    equivalent = True
    margin = float(config["A3"]["equivalence_margin"])
    gate = config["A3"]["format_effect"]
    for index, (left, right) in enumerate(itertools.combinations(A3_CONDITIONS, 2)):
        seed_values = [
            _accuracy(rows, seed=seed, experiment="A3", condition=left)
            - _accuracy(rows, seed=seed, experiment="A3", condition=right)
            for seed in seeds
        ]
        summary = _summary(seed_values, config, offset=500 + index)
        scene_left, scene_right = _paired_condition_values(rows, "A3", left, right)
        pair_sensitive = _format_sensitive(summary, gate)
        interval = summary["confidence_interval"]
        pair_equivalent = interval["lower"] >= -margin and interval["upper"] <= margin
        sensitive = sensitive or pair_sensitive
        equivalent = equivalent and pair_equivalent
        pairwise[f"{left}_minus_{right}"] = {
            "summary": summary,
            "paired_scene_test": paired_bootstrap_test(
                scene_left,
                scene_right,
                alternative="two-sided",
                samples=int(config["statistics"]["bootstrap_samples"]),
                seed=int(config["statistics"]["statistics_seed"]) + 600 + index,
            ),
            "format_sensitive": pair_sensitive,
            "equivalent_within_margin": pair_equivalent,
        }
    classification = (
        "FORMAT_SENSITIVE" if sensitive else "FORMAT_INVARIANT" if equivalent else "INCONCLUSIVE"
    )
    return {
        "conditions": conditions,
        "pairwise": pairwise,
        "equivalence_margin": margin,
        "format_classification": classification,
    }


def _availability(summary: dict[str, Any], config: dict[str, Any]) -> str:
    high = config["A1"]["high_availability"]
    low = config["A1"]["low_availability"]
    interval = summary["confidence_interval"]
    chance = float(config["A1"]["chance_accuracy"])
    seed_fraction = sum(value > chance for value in summary["values"]) / len(summary["values"])
    if (
        summary["mean"] >= float(high["minimum_mean_accuracy"])
        and interval["lower"] >= float(high["minimum_ci_lower"])
        and seed_fraction >= float(high["minimum_seed_fraction_above_chance"])
    ):
        return "HIGH"
    if (
        summary["mean"] <= float(low["maximum_mean_accuracy"])
        and interval["upper"] <= float(low["maximum_ci_upper"])
    ):
        return "LOW"
    return "INDETERMINATE"


def _stable_positive(summary: dict[str, Any], gate: dict[str, Any]) -> bool:
    return bool(
        summary["mean"] >= float(gate["minimum_gain"])
        and (
            not bool(gate["require_ci_lower_above_zero"])
            or summary["confidence_interval"]["lower"] > 0.0
        )
        and summary["positive_seed_fraction"]
        >= float(gate["minimum_positive_seed_fraction"])
    )


def _format_sensitive(summary: dict[str, Any], gate: dict[str, Any]) -> bool:
    mean = float(summary["mean"])
    interval = summary["confidence_interval"]
    consistent = (
        sum(value > 0 for value in summary["values"]) / len(summary["values"])
        if mean > 0
        else sum(value < 0 for value in summary["values"]) / len(summary["values"])
    )
    excludes_zero = interval["lower"] > 0.0 or interval["upper"] < 0.0
    return bool(
        abs(mean) >= float(gate["minimum_absolute_difference"])
        and (not bool(gate["require_ci_excludes_zero"]) or excludes_zero)
        and consistent >= float(gate["minimum_consistent_seed_fraction"])
    )


def _summary(values: list[float], config: dict[str, Any], *, offset: int) -> dict[str, Any]:
    return multiple_seed_summary(
        values,
        confidence=float(config["statistics"]["confidence_level"]),
        bootstrap_samples=int(config["statistics"]["bootstrap_samples"]),
        seed=int(config["statistics"]["statistics_seed"]) + offset,
        significance_test="paired_bootstrap",
    )


def _accuracy(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    experiment: str,
    condition: str | None = None,
    task: str | None = None,
) -> float:
    selected = [
        row
        for row in rows
        if int(row["seed"]) == seed
        and row["experiment"] == experiment
        and (condition is None or row["condition"] == condition)
        and (task is None or row["task"] == task)
    ]
    if not selected:
        raise ValueError(f"missing rows for {seed=} {experiment=} {condition=} {task=}")
    return statistics.fmean(float(bool(row["correct"])) for row in selected)


def _count(
    rows: list[dict[str, Any]], *, seed: int, experiment: str, task: str
) -> int:
    return sum(
        int(row["seed"]) == seed and row["experiment"] == experiment and row["task"] == task
        for row in rows
    )


def _slice_accuracies(
    rows: list[dict[str, Any]], experiment: str, condition: str, field: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        if row["experiment"] == experiment and row["condition"] == condition:
            groups[str(row[field])].append(bool(row["correct"]))
    return {
        key: {"accuracy": statistics.fmean(values), "n": len(values)}
        for key, values in sorted(groups.items())
    }


def _a2_seed_effect(
    rows: list[dict[str, Any]], *, seed: int, formula: tuple[str, str]
) -> float:
    values = {
        condition: _accuracy(rows, seed=seed, experiment="A2", condition=condition)
        for condition in A2_CONDITIONS
    }
    left, right = formula
    if right == "max_component":
        return values[left] - max(values["object_evidence"], values["relation_evidence"])
    if right == "additive_components":
        return (
            values["complete_evidence"]
            - values["object_evidence"]
            - values["relation_evidence"]
            + values["image_only"]
        )
    return values[left] - values[right]


def _a2_scene_pairs(
    rows: list[dict[str, Any]], *, formula: tuple[str, str]
) -> tuple[list[float], list[float]]:
    grouped = _scene_condition_map(rows, "A2")
    left_values: list[float] = []
    right_values: list[float] = []
    left, right = formula
    for values in grouped.values():
        if right == "max_component":
            left_values.append(values["complete_evidence"])
            right_values.append(max(values["object_evidence"], values["relation_evidence"]))
        elif right == "additive_components":
            left_values.append(values["complete_evidence"] + values["image_only"])
            right_values.append(values["object_evidence"] + values["relation_evidence"])
        else:
            left_values.append(values[left])
            right_values.append(values[right])
    return left_values, right_values


def _paired_condition_values(
    rows: list[dict[str, Any]], experiment: str, left: str, right: str
) -> tuple[list[float], list[float]]:
    grouped = _scene_condition_map(rows, experiment)
    return (
        [values[left] for values in grouped.values()],
        [values[right] for values in grouped.values()],
    )


def _scene_condition_map(
    rows: list[dict[str, Any]], experiment: str
) -> dict[tuple[int, str], dict[str, float]]:
    grouped: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["experiment"] == experiment:
            grouped[(int(row["seed"]), str(row["source_scene_id"]))][str(row["condition"])] = float(
                bool(row["correct"])
            )
    return dict(sorted(grouped.items()))


def _hypothesis_a_status(statuses: dict[str, str]) -> str:
    if all(value == "HIGH" for value in statuses.values()):
        return "FALSIFIED"
    if any(value == "LOW" for value in statuses.values()):
        return "SUPPORTED_IN_SCOPE"
    return "INCONCLUSIVE"


def _hypothesis_b_status(
    format_status: str, stable: dict[str, bool], classification: str
) -> str:
    if classification in {
        "RELATIONAL_GROUNDING_FAILURE",
        "FORMAT_CONDITIONED_RELATIONAL_INTEGRATION_FAILURE",
        "COMPOSITIONAL_REASONING_BOTTLENECK",
    }:
        return "SUPPORTED_OR_REFINED_IN_SCOPE"
    if format_status == "FORMAT_INVARIANT" and stable["relation_contribution"]:
        return "PARTIALLY_FALSIFIED"
    return "INCONCLUSIVE"


__all__ = ["analyze_predictions", "classify_mechanism"]
