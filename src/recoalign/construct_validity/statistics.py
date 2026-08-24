"""Task-specific hierarchical inference for PIVOT_EXP_A3."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from statistics import NormalDist
from typing import Any

from .trial_builder import TASKS


def analyze_construct_predictions(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    primary = [row for row in rows if row["response_method"] == "forced_choice"]
    secondary = [row for row in rows if row["response_method"] == "free_generation"]
    gate_a = _gate_a(primary, config)
    candidate_results = {}
    for manipulation in ("M1", "M2"):
        selected_primary = [row for row in primary if row["manipulation"] == manipulation]
        selected_secondary = [row for row in secondary if row["manipulation"] == manipulation]
        gate_a_comprehension = _contract_comprehension_gate(
            selected_primary, config, offset=50
        )
        gate_b = _gate_b(selected_secondary, config)
        gate_c = _sufficiency_gate(
            selected_primary,
            config,
            construct="CV2",
            mean_key="per_task_mean_minimum",
            lower_key="per_task_confidence_interval_lower_minimum",
            gate_name="C_scaffold_comprehension",
            offset=100,
        )
        gate_d = _sufficiency_gate(
            selected_primary,
            config,
            construct="CV3",
            mean_key="per_task_mean_minimum",
            lower_key="per_task_confidence_interval_lower_minimum",
            gate_name="D_scene_semantic_sufficiency",
            offset=200,
        )
        gate_e = _separation_gate(selected_primary, config, offset=300)
        gate_f = _interference_gate(selected_primary, config, offset=400)
        gates = {
            "A": bool(gate_a["passed"] and gate_a_comprehension["passed"]),
            "B": bool(gate_b["passed"]),
            "C": bool(gate_c["passed"]),
            "D": bool(gate_d["passed"]),
            "E": bool(gate_e["passed"]),
            "F": bool(gate_f["passed"]),
        }
        candidate_results[manipulation] = {
            "passed_all_gates": all(gates.values()),
            "gates": gates,
            "gate_A": {
                "passed": gates["A"],
                "measurement_validity": gate_a,
                "CV1_comprehension": gate_a_comprehension,
            },
            "gate_B": gate_b,
            "gate_C": gate_c,
            "gate_D": gate_d,
            "gate_E": gate_e,
            "gate_F": gate_f,
        }
    anchor = _anchor_summary(primary, config)
    return {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A3",
        "prediction_count": len(rows),
        "forced_choice_count": len(primary),
        "free_generation_count": len(secondary),
        "completed_seeds": sorted({int(row["seed"]) for row in rows}),
        "gate_A": gate_a,
        "candidates": candidate_results,
        "M0_historical_anchor": anchor,
        "primary_endpoints": [
            "answer_contract_validity",
            "shape_sufficiency",
            "color_sufficiency",
            "object_identity_sufficiency",
            "entity_attribute_binding_sufficiency",
        ],
        "aggregate_override_allowed": False,
    }


def hierarchical_summary(
    values: dict[tuple[int, str], float],
    config: dict[str, Any],
    *,
    offset: int,
) -> dict[str, Any]:
    if not values:
        raise ValueError("hierarchical summary requires paired seed-scene observations")
    by_seed: dict[int, dict[str, float]] = defaultdict(dict)
    for (seed, scene_id), value in values.items():
        by_seed[int(seed)][str(scene_id)] = float(value)
    observed = statistics.fmean(
        value for scene_values in by_seed.values() for value in scene_values.values()
    )
    samples = int(config["statistics"]["bootstrap_samples"])
    base_seed = int(config["statistics"]["bootstrap_seed"])
    rng = random.Random(base_seed + offset)
    seeds = sorted(by_seed)
    bootstrap = []
    for _ in range(samples):
        sampled_seeds = [rng.choice(seeds) for _ in seeds]
        replicate = []
        for seed in sampled_seeds:
            scene_values = list(by_seed[seed].values())
            replicate.extend(rng.choice(scene_values) for _ in scene_values)
        bootstrap.append(statistics.fmean(replicate))
    confidence = float(config["statistics"]["confidence_level"])
    lower_probability = (1.0 - confidence) / 2.0
    seed_means = {
        seed: statistics.fmean(scene_values.values())
        for seed, scene_values in sorted(by_seed.items())
    }
    return {
        "mean": observed,
        "confidence_interval": {
            "level": confidence,
            "lower": _quantile(bootstrap, lower_probability),
            "upper": _quantile(bootstrap, 1.0 - lower_probability),
            "method": "hierarchical_seed_scene_percentile_bootstrap",
            "samples": samples,
            "seed": base_seed + offset,
        },
        "seed_means": seed_means,
        "n_scenes": len(values),
        "n_seeds": len(by_seed),
    }


def paired_tost(values: list[float], *, margin: float, alpha: float) -> dict[str, Any]:
    if len(values) < 2 or margin <= 0:
        raise ValueError("paired TOST requires at least two observations and positive margin")
    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    normal = NormalDist()
    if standard_error == 0.0:
        p_lower = 0.0 if mean > -margin else 1.0
        p_upper = 0.0 if mean < margin else 1.0
        interval = (mean, mean)
    else:
        p_lower = 1.0 - normal.cdf((mean + margin) / standard_error)
        p_upper = normal.cdf((mean - margin) / standard_error)
        critical = normal.inv_cdf(1.0 - alpha)
        interval = (mean - critical * standard_error, mean + critical * standard_error)
    return {
        "mean": mean,
        "margin": [-margin, margin],
        "alpha": alpha,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_max": max(p_lower, p_upper),
        "equivalence_interval": {"lower": interval[0], "upper": interval[1]},
        "equivalent_unadjusted": bool(
            p_lower <= alpha
            and p_upper <= alpha
            and interval[0] > -margin
            and interval[1] < margin
        ),
        "method": "paired_normal_TOST",
    }


def hierarchical_tost(
    values: dict[tuple[int, str], float],
    config: dict[str, Any],
    *,
    margin: float,
    alpha: float,
    offset: int,
) -> dict[str, Any]:
    if len(values) < 2 or margin <= 0:
        raise ValueError("hierarchical TOST requires paired observations and positive margin")
    by_seed: dict[int, list[float]] = defaultdict(list)
    for (seed, _scene_id), value in values.items():
        by_seed[int(seed)].append(float(value))
    seeds = sorted(by_seed)
    observed = statistics.fmean(value for rows in by_seed.values() for value in rows)
    samples = int(config["statistics"]["bootstrap_samples"])
    rng = random.Random(int(config["statistics"]["bootstrap_seed"]) + offset)
    bootstrap = []
    for _ in range(samples):
        selected_seeds = [rng.choice(seeds) for _ in seeds]
        replicate = []
        for seed in selected_seeds:
            scene_values = by_seed[seed]
            replicate.extend(rng.choice(scene_values) for _ in scene_values)
        bootstrap.append(statistics.fmean(replicate))
    centered = [value - observed for value in bootstrap]
    p_lower = (sum(value >= observed + margin for value in centered) + 1) / (samples + 1)
    p_upper = (sum(value <= observed - margin for value in centered) + 1) / (samples + 1)
    interval = {
        "lower": _quantile(bootstrap, alpha),
        "upper": _quantile(bootstrap, 1.0 - alpha),
        "level": 1.0 - 2.0 * alpha,
    }
    return {
        "mean": observed,
        "margin": [-margin, margin],
        "alpha": alpha,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_max": max(p_lower, p_upper),
        "equivalence_interval": interval,
        "equivalent_unadjusted": bool(
            p_lower <= alpha
            and p_upper <= alpha
            and interval["lower"] > -margin
            and interval["upper"] < margin
        ),
        "method": "hierarchical_seed_scene_percentile_bootstrap_TOST",
        "bootstrap_samples": samples,
        "bootstrap_seed": int(config["statistics"]["bootstrap_seed"]) + offset,
        "n_pairs": len(values),
        "n_seeds": len(by_seed),
    }


def holm_correction(p_values: dict[str, float], *, alpha: float) -> dict[str, Any]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    output = {}
    previous_adjusted = 0.0
    continue_rejecting = True
    total = len(ordered)
    for rank, (name, p_value) in enumerate(ordered, start=1):
        multiplier = total - rank + 1
        threshold = alpha / multiplier
        adjusted = max(previous_adjusted, min(1.0, multiplier * float(p_value)))
        reject = continue_rejecting and float(p_value) <= threshold
        if not reject:
            continue_rejecting = False
        output[name] = {
            "raw_p": float(p_value),
            "adjusted_p": adjusted,
            "threshold": threshold,
            "reject": reject,
            "rank": rank,
        }
        previous_adjusted = adjusted
    return output


def _gate_a(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    valid = sum(bool(row.get("valid_measurement")) for row in rows)
    rate = valid / len(rows) if rows else 0.0
    required = float(config["gates"]["A_primary_answer_validity"]["valid_measurement_rate"])
    return {
        "passed": bool(rows) and rate == required,
        "valid_measurements": valid,
        "n": len(rows),
        "valid_measurement_rate": rate,
        "required": required,
    }


def _gate_b(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    parsed = sum(bool(row.get("parsed")) for row in rows)
    n = len(rows)
    rate = parsed / n if n else 0.0
    lower, upper = wilson_interval(parsed, n) if n else (0.0, 1.0)
    gate = config["gates"]["B_secondary_parser_integrity"]
    passed = bool(
        n
        and rate >= float(gate["parse_rate_minimum"])
        and lower >= float(gate["confidence_interval_lower_minimum"])
    )
    return {
        "passed": passed,
        "parsed": parsed,
        "n": n,
        "parse_rate": rate,
        "confidence_interval": {"lower": lower, "upper": upper, "method": "Wilson"},
    }


def _contract_comprehension_gate(
    rows: list[dict[str, Any]], config: dict[str, Any], *, offset: int
) -> dict[str, Any]:
    gate = config["gates"]["A_primary_answer_validity"]
    by_condition = {}
    for index, evidence_truth in enumerate(("oracle", "corrupted")):
        selected = [
            row
            for row in rows
            if row["construct"] == "CV1" and row["evidence_truth"] == evidence_truth
        ]
        values = {
            (int(row["seed"]), str(row["scene_id"])): float(bool(row["correct"]))
            for row in selected
        }
        summary = hierarchical_summary(values, config, offset=offset + index)
        seed_passes = sum(
            value >= float(gate["CV1_per_condition_mean_minimum"])
            for value in summary["seed_means"].values()
        )
        summary["seed_replication"] = {
            "passing_seeds": seed_passes,
            "required": int(config["statistics"]["seed_replication"]["minimum_passing_seeds"]),
        }
        summary["passed"] = bool(
            summary["mean"] >= float(gate["CV1_per_condition_mean_minimum"])
            and summary["confidence_interval"]["lower"]
            >= float(gate["CV1_per_condition_confidence_interval_lower_minimum"])
            and seed_passes
            >= int(config["statistics"]["seed_replication"]["minimum_passing_seeds"])
        )
        by_condition[evidence_truth] = summary
    return {
        "passed": all(bool(value["passed"]) for value in by_condition.values()),
        "ground_truth_policy": "active_scaffold_declaration",
        "by_condition": by_condition,
    }


def _sufficiency_gate(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    construct: str,
    mean_key: str,
    lower_key: str,
    gate_name: str,
    offset: int,
) -> dict[str, Any]:
    gate = config["gates"][gate_name]
    by_task = {}
    for task_index, task in enumerate(TASKS):
        selected = [
            row
            for row in rows
            if row["construct"] == construct
            and row["task"] == task
            and row["evidence_truth"] == "oracle"
        ]
        values = {
            (int(row["seed"]), str(row["scene_id"])): float(bool(row["correct"]))
            for row in selected
        }
        summary = hierarchical_summary(values, config, offset=offset + task_index)
        seed_passes = sum(
            value >= float(gate[mean_key]) for value in summary["seed_means"].values()
        )
        summary["seed_replication"] = {
            "passing_seeds": seed_passes,
            "required": int(config["statistics"]["seed_replication"]["minimum_passing_seeds"]),
        }
        summary["passed"] = bool(
            summary["mean"] >= float(gate[mean_key])
            and summary["confidence_interval"]["lower"] >= float(gate[lower_key])
            and seed_passes
            >= int(config["statistics"]["seed_replication"]["minimum_passing_seeds"])
        )
        by_task[task] = summary
    return {"passed": all(bool(value["passed"]) for value in by_task.values()), "by_task": by_task}


def _separation_gate(
    rows: list[dict[str, Any]], config: dict[str, Any], *, offset: int
) -> dict[str, Any]:
    gate = config["gates"]["E_intervention_separation"]
    alpha = float(config["statistics"]["alpha"])
    by_task = {}
    p_values = {}
    for task_index, task in enumerate(TASKS):
        selected = [row for row in rows if row["construct"] == "CV3" and row["task"] == task]
        paired: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
        for row in selected:
            paired[(int(row["seed"]), str(row["scene_id"]))][str(row["evidence_truth"])] = float(
                bool(row["correct"])
            )
        values = {
            key: cells["oracle"] - cells["corrupted"]
            for key, cells in paired.items()
            if set(cells) == {"oracle", "corrupted"}
        }
        summary = hierarchical_summary(values, config, offset=offset + task_index)
        p_value = _bootstrap_positive_p(summary, values, config, offset=offset + 20 + task_index)
        p_values[task] = p_value
        seed_passes = sum(
            value >= float(gate["per_task_oracle_minus_corrupted_minimum"])
            for value in summary["seed_means"].values()
        )
        corrupted_accuracy = statistics.fmean(
            cells["corrupted"] for cells in paired.values() if "corrupted" in cells
        )
        summary.update(
            {
                "raw_p_one_sided": p_value,
                "corrupted_accuracy": corrupted_accuracy,
                "seed_replication": {
                    "passing_seeds": seed_passes,
                    "required": int(
                        config["statistics"]["seed_replication"]["minimum_passing_seeds"]
                    ),
                },
            }
        )
        by_task[task] = summary
    holm = holm_correction(p_values, alpha=alpha)
    for task in TASKS:
        summary = by_task[task]
        summary["holm"] = holm[task]
        summary["passed"] = bool(
            summary["mean"] >= float(gate["per_task_oracle_minus_corrupted_minimum"])
            and summary["confidence_interval"]["lower"]
            > float(gate["per_task_confidence_interval_lower_strictly_above"])
            and summary["corrupted_accuracy"]
            < float(config["gates"]["D_scene_semantic_sufficiency"]["per_task_mean_minimum"])
            and bool(summary["holm"]["reject"])
            and summary["seed_replication"]["passing_seeds"]
            >= summary["seed_replication"]["required"]
        )
    return {
        "passed": all(bool(value["passed"]) for value in by_task.values()),
        "holm_family": holm,
        "by_task": by_task,
    }


def _interference_gate(
    rows: list[dict[str, Any]], config: dict[str, Any], *, offset: int
) -> dict[str, Any]:
    margin = float(config["gates"]["F_image_interference"]["paired_TOST_margin"])
    alpha = float(config["statistics"]["alpha"])
    by_task = {}
    p_values = {}
    for task_index, task in enumerate(TASKS):
        selected = [
            row
            for row in rows
            if row["task"] == task and row["evidence_truth"] == "oracle"
        ]
        paired: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
        for row in selected:
            paired[(int(row["seed"]), str(row["scene_id"]))][str(row["image_context"])] = float(
                bool(row["correct"])
            )
        differences = {
            key: cells["neutral_image"] - cells["original_scene_image"]
            for key, cells in paired.items()
            if set(cells) == {"neutral_image", "original_scene_image"}
        }
        tost = hierarchical_tost(
            differences,
            config,
            margin=margin,
            alpha=alpha,
            offset=offset + task_index,
        )
        p_values[task] = float(tost["p_max"])
        by_task[task] = tost
    holm = holm_correction(p_values, alpha=alpha)
    for task in TASKS:
        by_task[task]["holm"] = holm[task]
        interval = by_task[task]["equivalence_interval"]
        by_task[task]["passed"] = bool(
            holm[task]["reject"]
            and interval["lower"] > -margin
            and interval["upper"] < margin
        )
    return {
        "passed": all(bool(value["passed"]) for value in by_task.values()),
        "holm_family": holm,
        "by_task": by_task,
        "non_significance_is_equivalence": False,
    }


def _anchor_summary(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["manipulation"] == "M0"
        and row["construct"] == "CV3"
        and row["evidence_truth"] == "oracle"
    ]
    by_task = {}
    for index, task in enumerate(TASKS):
        values = {
            (int(row["seed"]), str(row["scene_id"])): float(bool(row["correct"]))
            for row in selected
            if row["task"] == task
        }
        by_task[task] = hierarchical_summary(values, config, offset=900 + index)
    return {"selection_eligible": False, "by_task": by_task}


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("Wilson interval requires 0 <= successes <= n and n > 0")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / n
    denominator = 1.0 + z * z / n
    center = (proportion + z * z / (2.0 * n)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n))
        / denominator
    )
    return center - radius, center + radius


def _bootstrap_positive_p(
    summary: dict[str, Any],
    values: dict[tuple[int, str], float],
    config: dict[str, Any],
    *,
    offset: int,
) -> float:
    observed = float(summary["mean"])
    by_seed: dict[int, list[float]] = defaultdict(list)
    for (seed, _scene_id), value in values.items():
        by_seed[int(seed)].append(float(value))
    seeds = sorted(by_seed)
    samples = int(config["statistics"]["bootstrap_samples"])
    rng = random.Random(int(config["statistics"]["bootstrap_seed"]) + offset)
    exceedances = 0
    for _ in range(samples):
        sampled_seeds = [rng.choice(seeds) for _ in seeds]
        replicate = []
        for seed in sampled_seeds:
            scene_values = by_seed[seed]
            replicate.extend(rng.choice(scene_values) for _ in scene_values)
        centered = statistics.fmean(replicate) - observed
        exceedances += centered >= observed
    return (exceedances + 1) / (samples + 1)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


__all__ = [
    "analyze_construct_predictions",
    "hierarchical_summary",
    "hierarchical_tost",
    "holm_correction",
    "paired_tost",
    "wilson_interval",
]
