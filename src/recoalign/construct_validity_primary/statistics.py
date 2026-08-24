"""Task-specific hierarchical inference for the primary-only study."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any

TASKS = ("shape", "color", "object_identity", "entity_attribute_binding")


def analyze_primary_predictions(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    gate_a_measurement = _measurement_gate(rows)
    candidates: dict[str, Any] = {}
    for candidate_index, manipulation in enumerate(("M1", "M2")):
        selected = [row for row in rows if row["manipulation"] == manipulation]
        cv1 = _cv1_gate(selected, config, offset=50 + candidate_index * 10)
        gate_c = _sufficiency_gate(
            selected,
            config,
            construct="CV2",
            gate_name="C_scaffold_comprehension",
            offset=100 + candidate_index * 20,
        )
        gate_d = _sufficiency_gate(
            selected,
            config,
            construct="CV3",
            gate_name="D_scene_semantic_sufficiency",
            offset=200 + candidate_index * 20,
        )
        gate_e = _separation_gate(selected, config, offset=300 + candidate_index * 40)
        gate_f = _interference_gate(selected, config, offset=400 + candidate_index * 40)
        gates = {
            "A": bool(gate_a_measurement["passed"] and cv1["passed"]),
            "C": bool(gate_c["passed"]),
            "D": bool(gate_d["passed"]),
            "E": bool(gate_e["passed"]),
            "F": bool(gate_f["passed"]),
        }
        candidates[manipulation] = {
            "gates": gates,
            "passed_all_gates": all(gates.values()),
            "gate_A": {
                "passed": gates["A"],
                "measurement_validity": gate_a_measurement,
                "CV1_comprehension": cv1,
            },
            "gate_C": gate_c,
            "gate_D": gate_d,
            "gate_E": gate_e,
            "gate_F": gate_f,
        }
    return {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A3P",
        "prediction_count": len(rows),
        "forced_choice_count": len(rows),
        "completed_seeds": sorted({int(row["seed"]) for row in rows}),
        "gate_A": gate_a_measurement,
        "secondary_gate": {
            "status": "RETIRED_BEFORE_VALIDATION",
            "participates_in_scientific_decision": False,
            "source_study": "PIVOT_EXP_A3R",
            "reason": "retired invalid external-validity instrument",
        },
        "candidates": candidates,
        "M0_historical_anchor": _anchor_summary(rows, config),
        "aggregate_override_allowed": False,
    }


def hierarchical_summary(
    values: dict[tuple[int, str], float], config: dict[str, Any], *, offset: int
) -> dict[str, Any]:
    if not values:
        raise ValueError("hierarchical summary requires seed-scene observations")
    by_seed: dict[int, dict[str, float]] = defaultdict(dict)
    for (seed, scene_id), value in values.items():
        if scene_id in by_seed[int(seed)]:
            raise ValueError("duplicate seed-scene observation")
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
    tail = (1.0 - confidence) / 2.0
    return {
        "mean": observed,
        "confidence_interval": {
            "level": confidence,
            "lower": quantile(bootstrap, tail),
            "upper": quantile(bootstrap, 1.0 - tail),
            "method": "hierarchical_seed_scene_percentile_bootstrap",
            "samples": samples,
            "seed": base_seed + offset,
        },
        "seed_means": {
            seed: statistics.fmean(scene_values.values())
            for seed, scene_values in sorted(by_seed.items())
        },
        "n_scenes": len(values),
        "n_seeds": len(by_seed),
    }


def paired_tost(values: list[float], *, margin: float, alpha: float) -> dict[str, Any]:
    """Synthetic-test reference implementation of paired normal TOST."""

    if len(values) < 2 or margin <= 0:
        raise ValueError("paired TOST requires at least two observations and positive margin")
    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    if standard_error == 0.0:
        p_lower = 0.0 if mean > -margin else 1.0
        p_upper = 0.0 if mean < margin else 1.0
        interval = (mean, mean)
    else:
        normal = statistics.NormalDist()
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
            p_lower <= alpha and p_upper <= alpha and interval[0] > -margin and interval[1] < margin
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
    bootstrap_seed = int(config["statistics"]["bootstrap_seed"]) + offset
    rng = random.Random(bootstrap_seed)
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
        "lower": quantile(bootstrap, alpha),
        "upper": quantile(bootstrap, 1.0 - alpha),
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
        "bootstrap_seed": bootstrap_seed,
        "n_pairs": len(values),
        "n_seeds": len(by_seed),
    }


def holm_correction(p_values: dict[str, float], *, alpha: float) -> dict[str, Any]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    output: dict[str, Any] = {}
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


def _measurement_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = 0
    four_score_rows = 0
    finite_rows = 0
    registered_rows = 0
    for row in rows:
        scores = row.get("choice_log_likelihoods")
        four = isinstance(scores, dict) and tuple(scores) == ("1", "2", "3", "4")
        finite = four and all(math.isfinite(float(value)) for value in scores.values())
        registered = str(row.get("selected_option_id")) in ("1", "2", "3", "4")
        four_score_rows += int(four)
        finite_rows += int(finite)
        registered_rows += int(registered)
        valid_rows += int(bool(row.get("valid_measurement")) and four and finite and registered)
    passed = (
        len(rows) == 27000
        and valid_rows == four_score_rows == finite_rows == registered_rows == 27000
    )
    return {
        "passed": passed,
        "n": len(rows),
        "valid_measurements": valid_rows,
        "valid_measurement_rate": valid_rows / len(rows) if rows else 0.0,
        "four_finite_score_rows": finite_rows,
        "registered_prediction_rows": registered_rows,
        "required": {
            "rows": 27000,
            "valid_measurement_rate": 1.0,
            "four_finite_scores_per_row": True,
            "registered_prediction_per_row": True,
        },
    }


def _cv1_gate(rows: list[dict[str, Any]], config: dict[str, Any], *, offset: int) -> dict[str, Any]:
    gate = config["gates"]["A_primary_answer_validity"]
    by_condition = {}
    for index, truth in enumerate(("oracle", "corrupted")):
        selected = [
            row for row in rows if row["construct"] == "CV1" and row["evidence_truth"] == truth
        ]
        summary = hierarchical_summary(_correctness_values(selected), config, offset=offset + index)
        _add_seed_replication(summary, float(gate["CV1_per_condition_mean_minimum"]), config)
        summary["passed"] = bool(
            summary["mean"] >= float(gate["CV1_per_condition_mean_minimum"])
            and summary["confidence_interval"]["lower"]
            >= float(gate["CV1_per_condition_confidence_interval_lower_minimum"])
            and _seed_replication_passed(summary)
        )
        by_condition[truth] = summary
    return {
        "passed": all(value["passed"] for value in by_condition.values()),
        "ground_truth_policy": "active_scaffold_declaration",
        "by_condition": by_condition,
    }


def _sufficiency_gate(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    construct: str,
    gate_name: str,
    offset: int,
) -> dict[str, Any]:
    gate = config["gates"][gate_name]
    mean_threshold = float(gate["per_task_mean_minimum"])
    lower_threshold = float(gate["per_task_confidence_interval_lower_minimum"])
    by_task = {}
    for index, task in enumerate(TASKS):
        selected = [
            row
            for row in rows
            if row["construct"] == construct
            and row["task"] == task
            and row["evidence_truth"] == "oracle"
        ]
        summary = hierarchical_summary(_correctness_values(selected), config, offset=offset + index)
        _add_seed_replication(summary, mean_threshold, config)
        summary["passed"] = bool(
            summary["mean"] >= mean_threshold
            and summary["confidence_interval"]["lower"] >= lower_threshold
            and _seed_replication_passed(summary)
        )
        by_task[task] = summary
    return {"passed": all(value["passed"] for value in by_task.values()), "by_task": by_task}


def _separation_gate(
    rows: list[dict[str, Any]], config: dict[str, Any], *, offset: int
) -> dict[str, Any]:
    gate = config["gates"]["E_intervention_separation"]
    gate_d = config["gates"]["D_scene_semantic_sufficiency"]
    alpha = float(config["statistics"]["alpha"])
    by_task: dict[str, Any] = {}
    p_values: dict[str, float] = {}
    for index, task in enumerate(TASKS):
        selected = [row for row in rows if row["construct"] == "CV3" and row["task"] == task]
        paired: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
        for row in selected:
            paired[(int(row["seed"]), str(row["scene_id"]))][str(row["evidence_truth"])] = float(
                bool(row["correctness"])
            )
        if (
            any(set(cells) != {"oracle", "corrupted"} for cells in paired.values())
            or len(paired) != 500
        ):
            raise ValueError(f"Gate E pairing incomplete for {task}")
        differences = {key: cells["oracle"] - cells["corrupted"] for key, cells in paired.items()}
        summary = hierarchical_summary(differences, config, offset=offset + index)
        p_value = _bootstrap_positive_p(summary, differences, config, offset=offset + 20 + index)
        p_values[task] = p_value
        corrupted_values = {key: cells["corrupted"] for key, cells in paired.items()}
        corrupted = hierarchical_summary(corrupted_values, config, offset=offset + 30 + index)
        _add_seed_replication(
            summary, float(gate["per_task_oracle_minus_corrupted_minimum"]), config
        )
        corrupted["would_independently_pass_gate_D"] = bool(
            corrupted["mean"] >= float(gate_d["per_task_mean_minimum"])
            and corrupted["confidence_interval"]["lower"]
            >= float(gate_d["per_task_confidence_interval_lower_minimum"])
        )
        summary.update(
            {
                "raw_p_one_sided": p_value,
                "corrupted_condition": corrupted,
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
            and not summary["corrupted_condition"]["would_independently_pass_gate_D"]
            and summary["holm"]["reject"]
            and _seed_replication_passed(summary)
        )
    return {
        "passed": all(value["passed"] for value in by_task.values()),
        "holm_family": holm,
        "by_task": by_task,
    }


def _interference_gate(
    rows: list[dict[str, Any]], config: dict[str, Any], *, offset: int
) -> dict[str, Any]:
    margin = float(config["gates"]["F_image_interference"]["paired_TOST_margin"])
    alpha = float(config["statistics"]["alpha"])
    by_task: dict[str, Any] = {}
    p_values: dict[str, float] = {}
    for index, task in enumerate(TASKS):
        selected = [
            row for row in rows if row["task"] == task and row["evidence_truth"] == "oracle"
        ]
        paired: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
        for row in selected:
            key = (int(row["seed"]), str(row["scene_id"]))
            paired[key][str(row["image_context"])] = float(bool(row["correctness"]))
        if (
            any(
                set(cells) != {"neutral_image", "original_scene_image"} for cells in paired.values()
            )
            or len(paired) != 500
        ):
            raise ValueError(f"Gate F pairing incomplete for {task}")
        differences = {
            key: cells["neutral_image"] - cells["original_scene_image"]
            for key, cells in paired.items()
        }
        tost = hierarchical_tost(
            differences, config, margin=margin, alpha=alpha, offset=offset + index
        )
        p_values[task] = float(tost["p_max"])
        by_task[task] = tost
    holm = holm_correction(p_values, alpha=alpha)
    for task in TASKS:
        interval = by_task[task]["equivalence_interval"]
        by_task[task]["holm"] = holm[task]
        by_task[task]["passed"] = bool(
            holm[task]["reject"] and interval["lower"] > -margin and interval["upper"] < margin
        )
    return {
        "passed": all(value["passed"] for value in by_task.values()),
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
    return {
        "selection_eligible": False,
        "descriptive_only": True,
        "by_task": {
            task: hierarchical_summary(
                _correctness_values([row for row in selected if row["task"] == task]),
                config,
                offset=900 + index,
            )
            for index, task in enumerate(TASKS)
        },
    }


def _correctness_values(rows: list[dict[str, Any]]) -> dict[tuple[int, str], float]:
    values = {
        (int(row["seed"]), str(row["scene_id"])): float(bool(row["correctness"])) for row in rows
    }
    if len(values) != len(rows):
        raise ValueError("duplicate seed-scene rows in task-specific endpoint")
    return values


def _add_seed_replication(
    summary: dict[str, Any], threshold: float, config: dict[str, Any]
) -> None:
    required = int(config["statistics"]["seed_replication"]["minimum_passing_seeds"])
    passing = sum(value >= threshold for value in summary["seed_means"].values())
    summary["seed_replication"] = {"passing_seeds": passing, "required": required}


def _seed_replication_passed(summary: dict[str, Any]) -> bool:
    replication = summary["seed_replication"]
    return int(replication["passing_seeds"]) >= int(replication["required"])


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
        exceedances += statistics.fmean(replicate) - observed >= observed
    return (exceedances + 1) / (samples + 1)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


__all__ = [
    "TASKS",
    "analyze_primary_predictions",
    "hierarchical_summary",
    "hierarchical_tost",
    "holm_correction",
    "paired_tost",
]
