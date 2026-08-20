"""Paired structural-sensitivity analyses for EXP002."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from evaluation.statistics import bootstrap_confidence_interval, paired_bootstrap_test

FULL = "scene_graph"
IMAGE_ONLY = "image_only"
RELATION_FLIP = "relation_flip"
OBJECT_SWAP = "object_swap"
RANDOM_GRAPH = "random_graph"
ORDER_RANDOMIZED = "graph_order_randomized"
LABEL_RANDOMIZED = "relation_label_randomized"

RELATION_GROUPS = {
    "left": "spatial",
    "right": "spatial",
    "above": "spatial",
    "below": "spatial",
    "front": "depth",
    "behind": "depth",
    "inside": "containment",
    "contains": "containment",
}


def partial_condition(ratio: float) -> str:
    return f"partial_graph_{int(round(ratio * 100)):02d}"


def analyze_seed(
    rows: list[dict[str, Any]],
    *,
    partial_ratios: Iterable[float],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Compute all preregistered paired effects and diagnostic slices for one seed."""

    ratios = tuple(float(value) for value in partial_ratios)
    conditions = _condition_summaries(rows, bootstrap_samples, seed)
    dependency: dict[str, Any] = {}
    for offset, ratio in enumerate(ratios):
        name = partial_condition(ratio)
        dependency[f"full_over_partial_{int(round(ratio * 100)):02d}"] = _paired_effect(
            rows,
            FULL,
            (name,),
            bootstrap_samples=bootstrap_samples,
            seed=seed + 101 + offset,
        )
    for offset, condition in enumerate((RELATION_FLIP, OBJECT_SWAP, RANDOM_GRAPH)):
        dependency[f"full_over_{condition}"] = _paired_effect(
            rows,
            FULL,
            (condition,),
            bootstrap_samples=bootstrap_samples,
            seed=seed + 201 + offset,
        )
    dependency["full_over_wrong_graph"] = _paired_effect(
        rows,
        FULL,
        (RELATION_FLIP, OBJECT_SWAP),
        bootstrap_samples=bootstrap_samples,
        seed=seed + 211,
    )
    dependency["full_over_relation_label_randomized"] = _paired_effect(
        rows,
        FULL,
        (LABEL_RANDOMIZED,),
        bootstrap_samples=bootstrap_samples,
        seed=seed + 212,
    )

    depths: dict[str, Any] = {}
    for depth in range(1, 5):
        sliced = [row for row in rows if int(row.get("hop_depth") or 0) == depth]
        depths[str(depth)] = {
            "n_scenes": len({str(row["scene_id"]) for row in sliced}),
            "conditions": _condition_summaries(sliced, bootstrap_samples, seed + depth * 13),
            "structural_dependency": {
                "full_over_random_graph": _paired_effect(
                    sliced,
                    FULL,
                    (RANDOM_GRAPH,),
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + 301 + depth,
                ),
                "full_over_partial_50": _paired_effect(
                    sliced,
                    FULL,
                    (partial_condition(0.5),),
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + 311 + depth,
                ),
            },
        }
    depth_dependency = {
        "full_over_random_graph": _depth_trend(depths, "full_over_random_graph"),
        "full_over_partial_50": _depth_trend(depths, "full_over_partial_50"),
    }

    relation_breakdown = _relation_breakdown(
        rows, bootstrap_samples=bootstrap_samples, seed=seed + 401
    )
    anti_shortcut = {
        "format_randomization": _paired_effect(
            rows,
            FULL,
            (ORDER_RANDOMIZED,),
            bootstrap_samples=bootstrap_samples,
            seed=seed + 501,
        ),
        "relation_label_randomization": dependency[
            "full_over_relation_label_randomized"
        ],
        "graph_length_matching": _length_control(rows),
    }
    return {
        "conditions": conditions,
        "structural_dependency": dependency,
        "reasoning_depth": depths,
        "depth_dependency": depth_dependency,
        "relation_breakdown": relation_breakdown,
        "anti_shortcut": anti_shortcut,
    }


def _condition_summaries(
    rows: list[dict[str, Any]], samples: int, seed: int
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(float(bool(row["correct"])))
    result: dict[str, Any] = {}
    for offset, (condition, values) in enumerate(sorted(grouped.items())):
        low, high = bootstrap_confidence_interval(
            values, samples=samples, seed=seed + offset
        )
        result[condition] = {
            "accuracy": sum(values) / len(values),
            "n": len(values),
            "ci95": [low, high],
        }
    return result


def _paired_effect(
    rows: list[dict[str, Any]],
    full_condition: str,
    controls: tuple[str, ...],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    values: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        values[str(row["scene_id"])][str(row["condition"])] = float(bool(row["correct"]))
    full: list[float] = []
    control: list[float] = []
    for scene_id in sorted(values):
        observed = values[scene_id]
        if full_condition not in observed or any(name not in observed for name in controls):
            continue
        full.append(observed[full_condition])
        control.append(sum(observed[name] for name in controls) / len(controls))
    if not full:
        return {
            "first": full_condition,
            "second": list(controls),
            "estimate": 0.0,
            "n": 0,
            "ci95": [0.0, 0.0],
            "statistical_test": None,
        }
    test = paired_bootstrap_test(
        full,
        control,
        alternative="greater",
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "first": full_condition,
        "second": list(controls),
        "estimate": sum(left - right for left, right in zip(full, control, strict=True))
        / len(full),
        "n": len(full),
        "ci95": [
            test["confidence_interval"]["lower"],
            test["confidence_interval"]["upper"],
        ],
        "statistical_test": test,
    }


def _depth_trend(depths: dict[str, Any], effect: str) -> dict[str, Any]:
    values = [
        float(depths[str(depth)]["structural_dependency"][effect]["estimate"])
        for depth in range(1, 5)
    ]
    slope = _linear_slope([1.0, 2.0, 3.0, 4.0], values)
    return {
        "values": {str(depth): values[depth - 1] for depth in range(1, 5)},
        "slope": slope,
        "depth4_minus_depth1": values[-1] - values[0],
        "positive_steps": sum(
            right > left for left, right in zip(values, values[1:], strict=False)
        ),
    }


def _relation_breakdown(
    rows: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    relations = sorted(
        {
            str(row.get("primary_relation"))
            for row in rows
            if str(row.get("primary_relation")) in RELATION_GROUPS
        }
    )
    exact: dict[str, Any] = {}
    for offset, relation in enumerate(relations):
        sliced = [row for row in rows if row.get("primary_relation") == relation]
        exact[relation] = {
            "conditions": _condition_summaries(
                sliced, bootstrap_samples, seed + offset * 7
            ),
            "full_over_random_graph": _paired_effect(
                sliced,
                FULL,
                (RANDOM_GRAPH,),
                bootstrap_samples=bootstrap_samples,
                seed=seed + 50 + offset,
            ),
        }
    groups: dict[str, Any] = {}
    for offset, group in enumerate(("spatial", "depth", "containment")):
        sliced = [row for row in rows if row.get("relation_group") == group]
        groups[group] = {
            "conditions": _condition_summaries(
                sliced, bootstrap_samples, seed + 100 + offset
            ),
            "full_over_random_graph": _paired_effect(
                sliced,
                FULL,
                (RANDOM_GRAPH,),
                bootstrap_samples=bootstrap_samples,
                seed=seed + 110 + offset,
            ),
        }
    return {"relations": exact, "groups": groups}


def _length_control(rows: list[dict[str, Any]]) -> dict[str, Any]:
    inputs: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        condition = str(row["condition"])
        if condition in {FULL, RANDOM_GRAPH}:
            inputs[str(row["scene_id"])][condition] = int(row["input"]["input_tokens"])
    deltas = [
        abs(values[FULL] - values[RANDOM_GRAPH])
        for values in inputs.values()
        if FULL in values and RANDOM_GRAPH in values
    ]
    return {
        "n_pairs": len(deltas),
        "max_absolute_token_delta": max(deltas, default=math.inf),
        "mean_absolute_token_delta": sum(deltas) / len(deltas) if deltas else math.inf,
        "passed": bool(deltas) and max(deltas) == 0,
    }


def _linear_slope(x: list[float], y: list[float]) -> float:
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    denominator = sum((value - mean_x) ** 2 for value in x)
    return (
        sum(
            (left - mean_x) * (right - mean_y)
            for left, right in zip(x, y, strict=True)
        )
        / denominator
    )


__all__ = [
    "FULL",
    "IMAGE_ONLY",
    "LABEL_RANDOMIZED",
    "OBJECT_SWAP",
    "ORDER_RANDOMIZED",
    "RANDOM_GRAPH",
    "RELATION_FLIP",
    "RELATION_GROUPS",
    "analyze_seed",
    "partial_condition",
]
