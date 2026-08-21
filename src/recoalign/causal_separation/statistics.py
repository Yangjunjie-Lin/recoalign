"""Scene-paired hierarchical inference, Holm control, and TOST equivalence."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from statistics import NormalDist
from typing import Any


def analyze_causal_predictions(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    main = [row for row in rows if row["family"] == "main"]
    manipulation = [row for row in rows if row["family"] == "manipulation_check"]
    scene_cells = _scene_cells(main)
    contrasts = {
        "E1_semantic_rescue": _contrast_by_scene(scene_cells, _e1),
        "E2_relation_under_oracle": _contrast_by_scene(scene_cells, _e2),
        "E3_json_minus_triples": _contrast_by_scene(scene_cells, _e3),
        "E4_semantic_x_relation": _contrast_by_scene(scene_cells, _e4),
        "E5_semantic_x_format": _contrast_by_scene(scene_cells, _e5),
        "E6_three_way": _contrast_by_scene(scene_cells, _e6),
    }
    summaries = {
        name: hierarchical_summary(values, config, offset=index)
        for index, (name, values) in enumerate(contrasts.items())
    }
    alpha = float(config["statistics"]["alpha"])
    margin = float(config["statistics"]["equivalence_margin"])
    summaries["E2_relation_under_oracle"]["tost"] = paired_tost(
        list(contrasts["E2_relation_under_oracle"].values()), margin=margin, alpha=alpha
    )
    summaries["E3_json_minus_triples"]["tost"] = paired_tost(
        list(contrasts["E3_json_minus_triples"].values()), margin=margin, alpha=alpha
    )
    holm = holm_correction(
        {
            name: summaries[name]["p_value_two_sided"]
            for name in ("E2_relation_under_oracle", "E3_json_minus_triples")
        },
        alpha=alpha,
    )
    for name, result in holm.items():
        summaries[name]["holm"] = result
    hop = _hop_summaries(main, config)
    manipulation_result = analyze_manipulation_check(manipulation, config)
    return {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A2",
        "prediction_count": len(rows),
        "main_prediction_count": len(main),
        "completed_seeds": sorted({int(row["seed"]) for row in rows}),
        "manipulation_check": manipulation_result,
        "estimands": summaries,
        "hop_depth": hop,
        "statistical_tests": {
            "holm_family": holm,
            "equivalence_margin": margin,
            "alpha": alpha,
            "bootstrap": "hierarchical_seed_scene_percentile",
        },
    }


def analyze_manipulation_check(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    grouped: dict[tuple[int, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        grouped[(int(row["seed"]), str(row["source_scene_id"]), str(row["task"]))][
            str(row["condition"])
        ] = float(bool(row["correct"]))
    oracle = {
        key: values["oracle_semantics"]
        for key, values in grouped.items()
        if set(values) == {"oracle_semantics", "corrupted_semantics"}
    }
    corrupted = {key: grouped[key]["corrupted_semantics"] for key in oracle}
    gain = {key: oracle[key] - corrupted[key] for key in oracle}
    oracle_summary = hierarchical_summary(oracle, config, offset=100)
    corrupted_summary = hierarchical_summary(corrupted, config, offset=101)
    gain_summary = hierarchical_summary(gain, config, offset=102)
    gate = config["manipulation_check"]
    passed = bool(
        oracle_summary["mean"] >= float(gate["oracle_minimum_mean_accuracy"])
        and oracle_summary["confidence_interval"]["lower"]
        >= float(gate["oracle_minimum_ci_lower"])
        and gain_summary["mean"] >= float(gate["minimum_oracle_minus_corrupted"])
        and gain_summary["confidence_interval"]["lower"] > 0.0
    )
    by_task = {}
    for task in sorted({key[2] for key in oracle}):
        task_oracle = {key: value for key, value in oracle.items() if key[2] == task}
        task_corrupted = {key: value for key, value in corrupted.items() if key[2] == task}
        by_task[task] = {
            "oracle_accuracy": statistics.fmean(task_oracle.values()),
            "corrupted_accuracy": statistics.fmean(task_corrupted.values()),
            "gain": statistics.fmean(task_oracle[key] - task_corrupted[key] for key in task_oracle),
            "n": len(task_oracle),
        }
    return {
        "passed": passed,
        "oracle": oracle_summary,
        "corrupted": corrupted_summary,
        "oracle_minus_corrupted": gain_summary,
        "by_task": by_task,
        "n_pairs": len(gain),
    }


def hierarchical_summary(
    values: dict[tuple[Any, ...], float], config: dict[str, Any], *, offset: int
) -> dict[str, Any]:
    if not values:
        raise ValueError("hierarchical summary requires observations")
    by_seed: dict[int, list[float]] = defaultdict(list)
    for key, value in values.items():
        by_seed[int(key[0])].append(float(value))
    seed_means = {seed: statistics.fmean(rows) for seed, rows in sorted(by_seed.items())}
    observed = statistics.fmean(value for rows in by_seed.values() for value in rows)
    samples = int(config["statistics"]["bootstrap_samples"])
    rng = random.Random(int(config["statistics"]["bootstrap_seed"]) + offset)
    seeds = sorted(by_seed)
    bootstrap: list[float] = []
    for _ in range(samples):
        selected_seeds = [rng.choice(seeds) for _ in seeds]
        replicate: list[float] = []
        for seed in selected_seeds:
            scene_values = by_seed[seed]
            replicate.extend(rng.choice(scene_values) for _ in scene_values)
        bootstrap.append(statistics.fmean(replicate))
    confidence = float(config["statistics"]["confidence_level"])
    lower = _quantile(bootstrap, (1.0 - confidence) / 2.0)
    upper = _quantile(bootstrap, 1.0 - (1.0 - confidence) / 2.0)
    negative_fraction = sum(value < 0 for value in seed_means.values()) / len(seed_means)
    positive_fraction = sum(value > 0 for value in seed_means.values()) / len(seed_means)
    centered = [value - observed for value in bootstrap]
    p_value = min(
        1.0,
        2.0
        * min(
            (sum(value <= -observed for value in centered) + 1) / (samples + 1),
            (sum(value >= -observed for value in centered) + 1) / (samples + 1),
        ),
    )
    return {
        "mean": observed,
        "confidence_interval": {
            "level": confidence,
            "lower": lower,
            "upper": upper,
            "method": "hierarchical_seed_scene_percentile_bootstrap",
            "samples": samples,
        },
        "seed_means": seed_means,
        "positive_seed_fraction": positive_fraction,
        "negative_seed_fraction": negative_fraction,
        "n_scenes": len(values),
        "n_seeds": len(by_seed),
        "p_value_two_sided": p_value,
    }


def paired_tost(values: list[float], *, margin: float, alpha: float) -> dict[str, Any]:
    if len(values) < 2 or margin <= 0:
        raise ValueError("TOST requires at least two paired values and a positive margin")
    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    normal = NormalDist()
    if standard_error == 0.0:
        lower_p = 0.0 if mean > -margin else 1.0
        upper_p = 0.0 if mean < margin else 1.0
        ci = (mean, mean)
    else:
        lower_z = (mean + margin) / standard_error
        upper_z = (mean - margin) / standard_error
        lower_p = 1.0 - normal.cdf(lower_z)
        upper_p = normal.cdf(upper_z)
        critical = normal.inv_cdf(1.0 - alpha)
        ci = (mean - critical * standard_error, mean + critical * standard_error)
    return {
        "margin": [-margin, margin],
        "alpha": alpha,
        "p_lower": lower_p,
        "p_upper": upper_p,
        "confidence_interval_90": {"lower": ci[0], "upper": ci[1]},
        "equivalent": bool(
            lower_p < alpha
            and upper_p < alpha
            and ci[0] > -margin
            and ci[1] < margin
        ),
        "method": "paired_normal_TOST",
    }


def holm_correction(p_values: dict[str, float], *, alpha: float) -> dict[str, Any]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    results: dict[str, Any] = {}
    previous_adjusted = 0.0
    continue_rejecting = True
    count = len(ordered)
    for rank, (name, p_value) in enumerate(ordered, start=1):
        threshold = alpha / (count - rank + 1)
        adjusted = max(previous_adjusted, min(1.0, (count - rank + 1) * p_value))
        reject = continue_rejecting and p_value <= threshold
        if not reject:
            continue_rejecting = False
        results[name] = {
            "raw_p": p_value,
            "adjusted_p": adjusted,
            "threshold": threshold,
            "reject_zero": reject,
            "rank": rank,
        }
        previous_adjusted = adjusted
    return results


def _scene_cells(
    rows: list[dict[str, Any]],
) -> dict[tuple[int, str], dict[tuple[str, str, str], float]]:
    grouped: dict[tuple[int, str], dict[tuple[str, str, str], float]] = defaultdict(dict)
    for row in rows:
        factors = row["factors"]
        key = (str(factors["semantic"]), str(factors["relation"]), str(factors["serialization"]))
        grouped[(int(row["seed"]), str(row["source_scene_id"]))][key] = float(
            bool(row["correct"])
        )
    if any(len(values) != 8 for values in grouped.values()):
        raise ValueError("main prediction matrix is incomplete")
    return dict(grouped)


def _contrast_by_scene(
    cells: dict[tuple[int, str], dict[tuple[str, str, str], float]],
    function: Any,
) -> dict[tuple[int, str], float]:
    return {key: float(function(values)) for key, values in cells.items()}


def _cell(
    values: dict[tuple[str, str, str], float],
    semantic: str,
    relation: str,
    fmt: str,
) -> float:
    return values[(semantic, relation, fmt)]


def _e1(values: dict[tuple[str, str, str], float]) -> float:
    contrasts = []
    for relation in ("correct_relation", "corrupted_relation"):
        for fmt in ("canonical_json", "canonical_triples"):
            contrasts.append(
                _cell(values, "oracle_semantics", relation, fmt)
                - _cell(values, "corrupted_semantics", relation, fmt)
            )
    return statistics.fmean(contrasts)


def _e2(values: dict[tuple[str, str, str], float]) -> float:
    return statistics.fmean(
        _cell(values, "oracle_semantics", "correct_relation", fmt)
        - _cell(values, "oracle_semantics", "corrupted_relation", fmt)
        for fmt in ("canonical_json", "canonical_triples")
    )


def _e3(values: dict[tuple[str, str, str], float]) -> float:
    return _cell(values, "oracle_semantics", "correct_relation", "canonical_json") - _cell(
        values, "oracle_semantics", "correct_relation", "canonical_triples"
    )


def _e4(values: dict[tuple[str, str, str], float]) -> float:
    return statistics.fmean(
        (
            _cell(values, "oracle_semantics", "correct_relation", fmt)
            - _cell(values, "oracle_semantics", "corrupted_relation", fmt)
        )
        - (
            _cell(values, "corrupted_semantics", "correct_relation", fmt)
            - _cell(values, "corrupted_semantics", "corrupted_relation", fmt)
        )
        for fmt in ("canonical_json", "canonical_triples")
    )


def _e5(values: dict[tuple[str, str, str], float]) -> float:
    oracle_format = _cell(
        values, "oracle_semantics", "correct_relation", "canonical_json"
    ) - _cell(values, "oracle_semantics", "correct_relation", "canonical_triples")
    corrupt_format = _cell(
        values, "corrupted_semantics", "correct_relation", "canonical_json"
    ) - _cell(values, "corrupted_semantics", "correct_relation", "canonical_triples")
    return oracle_format - corrupt_format


def _e6(values: dict[tuple[str, str, str], float]) -> float:
    def interaction(fmt: str) -> float:
        return (
            _cell(values, "oracle_semantics", "correct_relation", fmt)
            - _cell(values, "oracle_semantics", "corrupted_relation", fmt)
        ) - (
            _cell(values, "corrupted_semantics", "correct_relation", fmt)
            - _cell(values, "corrupted_semantics", "corrupted_relation", fmt)
        )

    return interaction("canonical_json") - interaction("canonical_triples")


def _hop_summaries(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for hop in (1, 2, 3, 4):
        selected = [row for row in rows if int(row["hop_depth"]) == hop]
        cells = _scene_cells(selected)
        output[str(hop)] = {
            name: hierarchical_summary(
                _contrast_by_scene(cells, function),
                config,
                offset=200 + hop * 10 + index,
            )
            for index, (name, function) in enumerate(
                (
                    ("E1_semantic_rescue", _e1),
                    ("E2_relation_under_oracle", _e2),
                    ("E3_json_minus_triples", _e3),
                    ("E4_semantic_x_relation", _e4),
                    ("E5_semantic_x_format", _e5),
                )
            )
        }
    return output


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
    "analyze_causal_predictions",
    "analyze_manipulation_check",
    "hierarchical_summary",
    "holm_correction",
    "paired_tost",
]
