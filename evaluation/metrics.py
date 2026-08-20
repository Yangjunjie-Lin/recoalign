"""Condition-level accuracy, paired contrasts, and JSON result serialization."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import jsonschema

from evaluation.statistics import bootstrap_confidence_interval, paired_t_test


def accuracy(values: Iterable[bool | int]) -> float:
    rows = [bool(value) for value in values]
    return sum(rows) / len(rows) if rows else 0.0


def bootstrap_ci(values: list[float], *, samples: int = 1000, seed: int = 7) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return bootstrap_confidence_interval(values, samples=max(1, samples), seed=seed)


def evaluate_rows(
    rows: Iterable[dict[str, Any]], *, bootstrap_samples: int = 1000, seed: int = 7
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(float(bool(row["correct"])))
    metrics: dict[str, Any] = {"conditions": {}, "n": sum(len(v) for v in grouped.values())}
    for condition, values in sorted(grouped.items()):
        low, high = bootstrap_ci(values, samples=bootstrap_samples, seed=seed + len(condition))
        metrics["conditions"][condition] = {
            "accuracy": sum(values) / len(values) if values else 0.0,
            "ci95": [low, high],
            "n": len(values),
        }
    return metrics


def paired_difference(
    rows: Iterable[dict[str, Any]],
    first: str,
    second: str,
    *,
    bootstrap_samples: int = 1000,
    seed: int = 7,
) -> dict[str, Any]:
    lookup: dict[tuple[str, str], bool] = {}
    for row in rows:
        lookup[(str(row["scene_id"]), str(row["condition"]))] = bool(row["correct"])
    scene_ids = sorted({scene_id for scene_id, _ in lookup})
    differences = [
        float(lookup[(scene_id, first)]) - float(lookup[(scene_id, second)])
        for scene_id in scene_ids
        if (scene_id, first) in lookup and (scene_id, second) in lookup
    ]
    result: dict[str, Any] = {
        "first": first,
        "second": second,
        "estimate": sum(differences) / len(differences) if differences else 0.0,
        "n": len(differences),
    }
    if differences:
        low, high = bootstrap_confidence_interval(
            differences, samples=bootstrap_samples, seed=seed
        )
        result["ci95"] = [low, high]
    else:
        result["ci95"] = [0.0, 0.0]
    if len(differences) >= 2:
        result["statistical_test"] = paired_t_test(
            differences, [0.0] * len(differences), alternative="greater"
        )
    else:
        result["statistical_test"] = None
    return result


def write_metrics(path: str | Path, payload: dict[str, Any]) -> None:
    schema_path = (
        Path(__file__).resolve().parents[1] / "schemas" / "structured_evaluation.schema.json"
    )
    if schema_path.exists():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.validate(payload, schema)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
