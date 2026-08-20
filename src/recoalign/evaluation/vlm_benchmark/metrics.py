"""Compositional, robustness, and group metrics from unified predictions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def compute_benchmark_metrics(
    predictions: list[dict[str, Any]], *, aggregation: str
) -> dict[str, Any]:
    if not predictions:
        raise ValueError("benchmark metrics require at least one prediction")
    correct = [bool(row["correct"]) for row in predictions]
    metrics: dict[str, Any] = {
        "accuracy": sum(correct) / len(correct),
        "failure_rate": 1.0 - sum(correct) / len(correct),
        "n": len(correct),
    }
    for field, prefix in (("dimension", "compositional"), ("category", "category")):
        grouped: dict[str, list[bool]] = defaultdict(list)
        for row in predictions:
            grouped[str(row[field])].append(bool(row["correct"]))
        metrics[prefix] = {
            name: {
                "accuracy": sum(values) / len(values),
                "failure_rate": 1.0 - sum(values) / len(values),
                "n": len(values),
            }
            for name, values in sorted(grouped.items())
        }
    if aggregation == "paired_group":
        groups: dict[str, list[bool]] = defaultdict(list)
        for row in predictions:
            groups[str(row["group_id"])].append(bool(row["correct"]))
        group_correct = [len(values) == 2 and all(values) for values in groups.values()]
        metrics["group_accuracy"] = sum(group_correct) / len(group_correct)
        metrics["group_failure_rate"] = 1.0 - metrics["group_accuracy"]
        metrics["groups"] = len(groups)
    distributions: dict[str, list[bool]] = defaultdict(list)
    for row in predictions:
        distribution = row.get("distribution")
        if distribution in {"iid", "ood"}:
            distributions[str(distribution)].append(bool(row["correct"]))
    if distributions:
        metrics["distribution"] = {
            name: {"accuracy": sum(values) / len(values), "n": len(values)}
            for name, values in sorted(distributions.items())
        }
        if distributions.get("iid") and distributions.get("ood"):
            iid = metrics["distribution"]["iid"]["accuracy"]
            ood = metrics["distribution"]["ood"]["accuracy"]
            metrics["generalization_gap"] = iid - ood
            metrics["ood_retention"] = ood / iid if iid else None
    return metrics


__all__ = ["compute_benchmark_metrics"]
