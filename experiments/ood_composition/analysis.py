"""Paired OOD, retention, depth, and anti-memorization analyses for EXP003."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from evaluation.statistics import bootstrap_confidence_interval, paired_bootstrap_test

IMAGE_ONLY = "image_only"
CAPTION = "caption"
GRAPH = "scene_graph"
RANDOM_GRAPH = "random_graph"
CONDITIONS = (IMAGE_ONLY, CAPTION, GRAPH, RANDOM_GRAPH)
OOD_SPLITS = ("composition_ood", "relation_ood", "hop_ood")


def analyze_seed(
    rows: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    """Compute all preregistered scalar gates and descriptive EXP003 slices."""

    test_rows = [row for row in rows if row["partition"] == "test"]
    iid = [row for row in test_rows if row["ood_split"] == "iid"]
    ood = [row for row in test_rows if row["ood_split"] in OOD_SPLITS]
    split_metrics: dict[str, Any] = {}
    for offset, name in enumerate(("iid", *OOD_SPLITS)):
        selected = [row for row in test_rows if row["ood_split"] == name]
        split_metrics[name] = {
            "conditions": _condition_summaries(
                selected, bootstrap_samples, seed + 101 * (offset + 1)
            ),
            "graph_over_caption": _paired_effect(
                selected,
                GRAPH,
                CAPTION,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 201 + offset,
            ),
            "graph_over_random_graph": _paired_effect(
                selected,
                GRAPH,
                RANDOM_GRAPH,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 211 + offset,
            ),
        }

    generalization: dict[str, Any] = {"conditions": {}}
    for offset, condition in enumerate(CONDITIONS):
        iid_accuracy = _accuracy([row for row in iid if row["condition"] == condition])
        ood_accuracy = _accuracy([row for row in ood if row["condition"] == condition])
        generalization["conditions"][condition] = {
            "iid_accuracy": iid_accuracy,
            "ood_accuracy": ood_accuracy,
            "generalization_gap": iid_accuracy - ood_accuracy,
            "ood_retention_ratio": ood_accuracy / iid_accuracy if iid_accuracy else 0.0,
            "paired_iid_over_ood": _paired_iid_ood(
                iid,
                ood,
                condition,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 301 + offset,
            ),
        }
    graph_retention = generalization["conditions"][GRAPH]["ood_retention_ratio"]
    caption_retention = generalization["conditions"][CAPTION]["ood_retention_ratio"]
    generalization["retention_advantage"] = {
        "graph_over_caption": graph_retention - caption_retention,
        "graph": graph_retention,
        "caption": caption_retention,
    }

    depth = _reasoning_depth(rows, bootstrap_samples=bootstrap_samples, seed=seed + 401)
    seen_unseen = _seen_unseen(rows, bootstrap_samples=bootstrap_samples, seed=seed + 501)
    anti_memorization = _anti_memorization(
        rows, bootstrap_samples=bootstrap_samples, seed=seed + 601
    )
    return {
        "conditions": _condition_summaries(test_rows, bootstrap_samples, seed),
        "splits": split_metrics,
        "ood": {
            "conditions": _condition_summaries(ood, bootstrap_samples, seed + 11),
            "graph_over_caption": _paired_effect(
                ood,
                GRAPH,
                CAPTION,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 21,
            ),
            "graph_over_random_graph": _paired_effect(
                ood,
                GRAPH,
                RANDOM_GRAPH,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 22,
            ),
        },
        "generalization": generalization,
        "reasoning_depth": depth,
        "seen_unseen_composition": seen_unseen,
        "anti_memorization": anti_memorization,
        "sample_counts": {
            "evaluated_scenes": len({str(row["scene_id"]) for row in rows}),
            "test_scenes": len({str(row["scene_id"]) for row in test_rows}),
            "iid_test_scenes": len({str(row["scene_id"]) for row in iid}),
            "ood_test_scenes": len({str(row["scene_id"]) for row in ood}),
            "predictions": len(rows),
        },
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
    first: str,
    second: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["scene_id"])][str(row["condition"])] = float(bool(row["correct"]))
    first_values: list[float] = []
    second_values: list[float] = []
    for scene_id in sorted(grouped):
        values = grouped[scene_id]
        if first in values and second in values:
            first_values.append(values[first])
            second_values.append(values[second])
    if not first_values:
        return {
            "first": first,
            "second": second,
            "estimate": 0.0,
            "n": 0,
            "ci95": [0.0, 0.0],
            "statistical_test": None,
        }
    test = paired_bootstrap_test(
        first_values,
        second_values,
        alternative="greater",
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "first": first,
        "second": second,
        "estimate": sum(
            left - right for left, right in zip(first_values, second_values, strict=True)
        )
        / len(first_values),
        "n": len(first_values),
        "ci95": [
            test["confidence_interval"]["lower"],
            test["confidence_interval"]["upper"],
        ],
        "statistical_test": test,
    }


def _paired_iid_ood(
    iid_rows: list[dict[str, Any]],
    ood_rows: list[dict[str, Any]],
    condition: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    iid_by_index = {
        _pair_index(row): float(bool(row["correct"]))
        for row in iid_rows
        if row["condition"] == condition
    }
    ood_by_index: dict[str, list[float]] = defaultdict(list)
    for row in ood_rows:
        if row["condition"] == condition:
            ood_by_index[_pair_index(row)].append(float(bool(row["correct"])))
    paired_iid: list[float] = []
    paired_ood: list[float] = []
    for index in sorted(iid_by_index):
        if index in ood_by_index:
            paired_iid.append(iid_by_index[index])
            paired_ood.append(sum(ood_by_index[index]) / len(ood_by_index[index]))
    if not paired_iid:
        return {"estimate": 0.0, "n": 0, "ci95": [0.0, 0.0], "statistical_test": None}
    test = paired_bootstrap_test(
        paired_iid,
        paired_ood,
        alternative="greater",
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "estimate": sum(
            left - right for left, right in zip(paired_iid, paired_ood, strict=True)
        )
        / len(paired_iid),
        "n": len(paired_iid),
        "ci95": [
            test["confidence_interval"]["lower"],
            test["confidence_interval"]["upper"],
        ],
        "statistical_test": test,
    }


def _reasoning_depth(
    rows: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    hop_rows = [row for row in rows if row["ood_split"] == "hop_ood"]
    depths: dict[str, Any] = {}
    advantages: list[float] = []
    for depth in range(1, 5):
        selected = [row for row in hop_rows if int(row["hop_depth"]) == depth]
        effect = _paired_effect(
            selected,
            GRAPH,
            CAPTION,
            bootstrap_samples=bootstrap_samples,
            seed=seed + depth,
        )
        advantages.append(float(effect["estimate"]))
        depths[str(depth)] = {
            "regime": "seen" if depth <= 2 else "unseen",
            "conditions": _condition_summaries(
                selected, bootstrap_samples, seed + 11 * depth
            ),
            "graph_over_caption": effect,
            "graph_over_random_graph": _paired_effect(
                selected,
                GRAPH,
                RANDOM_GRAPH,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 21 * depth,
            ),
        }
    return {
        "depths": depths,
        "graph_advantage_slope": _linear_slope([1.0, 2.0, 3.0, 4.0], advantages),
        "depth4_minus_depth1_advantage": advantages[-1] - advantages[0],
        "graph_advantage_by_depth": {
            str(depth): advantages[depth - 1] for depth in range(1, 5)
        },
    }


def _seen_unseen(
    rows: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    composition = [row for row in rows if row["ood_split"] == "composition_ood"]
    result: dict[str, Any] = {}
    for offset, partition in enumerate(("train", "test")):
        selected = [row for row in composition if row["partition"] == partition]
        result["seen" if partition == "train" else "unseen"] = {
            "conditions": _condition_summaries(
                selected, bootstrap_samples, seed + offset
            ),
            "graph_over_caption": _paired_effect(
                selected,
                GRAPH,
                CAPTION,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 11 + offset,
            ),
        }
    return result


def _anti_memorization(
    rows: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for offset, name in enumerate(
        ("object_identity_swap", "relation_recombination", "attribute_transfer")
    ):
        selected = [
            row
            for row in rows
            if row["partition"] == "test" and row.get("anti_memorization") == name
        ]
        result[name] = {
            "n_scenes": len({str(row["scene_id"]) for row in selected}),
            "conditions": _condition_summaries(
                selected, bootstrap_samples, seed + 17 * offset
            ),
            "graph_over_caption": _paired_effect(
                selected,
                GRAPH,
                CAPTION,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 31 + offset,
            ),
            "graph_over_random_graph": _paired_effect(
                selected,
                GRAPH,
                RANDOM_GRAPH,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 41 + offset,
            ),
        }
    return result


def _pair_index(row: dict[str, Any]) -> str:
    return str(row["pair_id"]).rsplit(":", 1)[-1]


def _accuracy(rows: list[dict[str, Any]]) -> float:
    return sum(bool(row["correct"]) for row in rows) / len(rows) if rows else 0.0


def _linear_slope(x: list[float], y: list[float]) -> float:
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    denominator = sum((value - mean_x) ** 2 for value in x)
    return sum(
        (left - mean_x) * (right - mean_y)
        for left, right in zip(x, y, strict=True)
    ) / denominator


__all__ = [
    "CAPTION",
    "CONDITIONS",
    "GRAPH",
    "IMAGE_ONLY",
    "OOD_SPLITS",
    "RANDOM_GRAPH",
    "analyze_seed",
]
