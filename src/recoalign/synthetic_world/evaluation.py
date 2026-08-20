"""Multi-seed evaluation entry point for the synthetic scientific instrument."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from experiments.runtime import build_vlm, run_conditions
from recoalign.research_registry import load_experiment_registry
from recoalign.synthetic_world.generator import GeneratorConfig, SyntheticWorldGenerator
from recoalign.synthetic_world.validation import validate_dataset


def seed_summary(values: list[float], confidence_level: float = 0.95) -> dict[str, Any]:
    """Summarize independent seed-level estimates with a Student-t 95% interval."""

    if not values:
        raise ValueError("at least one seed value is required")
    if confidence_level != 0.95:
        raise ValueError("the synthetic evaluator currently supports confidence_level=0.95")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    critical = _t_critical_95(len(values) - 1)
    half_width = critical * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95": [max(0.0, mean - half_width), min(1.0, mean + half_width)],
        "n_seeds": len(values),
        "per_seed": values,
    }


def evaluate_synthetic(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(config_path)
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("synthetic evaluation config must be a mapping")
    benchmark = config.get("benchmark")
    model = config.get("model")
    evaluation = config.get("evaluation")
    if not all(isinstance(value, dict) for value in (benchmark, model, evaluation)):
        raise ValueError("config requires benchmark, model, and evaluation mappings")
    benchmark = dict(benchmark)
    evaluation = dict(evaluation)
    critical = bool(evaluation.get("critical", False))
    required_seeds = 5 if critical else 3
    configured_seeds = evaluation.get("seeds")
    if configured_seeds is None:
        initial = int(benchmark.get("seed", 20260819))
        seeds = list(range(initial, initial + required_seeds))
    else:
        seeds = [int(value) for value in configured_seeds]
    if len(seeds) < required_seeds:
        raise ValueError(
            f"{'critical' if critical else 'default'} evaluation requires at least "
            f"{required_seeds} seeds"
        )
    count = int(benchmark.get("count", 120))
    split_strategy = str(benchmark.get("split_strategy", "composition"))
    root = Path(output_dir or benchmark.get("output_dir", "outputs/synthetic_benchmark"))
    root.mkdir(parents=True, exist_ok=True)
    conditions = tuple(
        str(value)
        for value in evaluation.get("conditions", ("caption", "scene_graph"))
    )
    if not conditions:
        raise ValueError("evaluation.conditions must be non-empty")
    evaluation_splits = tuple(str(value) for value in evaluation.get("splits", ()))

    all_predictions: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    integrity_failures: list[str] = []
    distribution = {
        "question_types": Counter(),
        "hop_depth": Counter(),
        "relations": Counter(),
    }
    for seed in seeds:
        generator = SyntheticWorldGenerator(
            GeneratorConfig(
                seed=seed,
                image_size=int(benchmark.get("resolution", 192)),
                write_images=bool(benchmark.get("write_images", True)),
                style=str(benchmark.get("style", "flat")),
                min_hops=int(benchmark.get("min_hops", 1)),
                max_hops=int(benchmark.get("max_hops", 4)),
            )
        )
        records = generator.generate(
            count,
            output_dir=root / "dataset" / f"seed_{seed}",
            seed=seed,
            split_strategy=split_strategy,
        )
        try:
            integrity = validate_dataset(
                records,
                split_strategy=split_strategy,
                check_images=bool(benchmark.get("write_images", True)),
            )
        except (ValueError, TypeError) as exc:
            integrity_failures.append(f"seed {seed}: {exc}")
            raise
        evaluated_records = (
            [record for record in records if record.split in evaluation_splits]
            if evaluation_splits
            else records
        )
        if not evaluated_records:
            raise ValueError(
                f"seed {seed}: no records match evaluation.splits={evaluation_splits}"
            )
        distribution["question_types"].update(
            str(record.metadata["question_type"]) for record in evaluated_records
        )
        distribution["hop_depth"].update(
            str(record.metadata["hop_depth"]) for record in evaluated_records
        )
        distribution["relations"].update(
            str(record.metadata["primary_relation"]) for record in evaluated_records
        )
        runtime_config = deepcopy(config)
        runtime_config["experiment"] = {
            "name": str(benchmark.get("name", "synthetic_compositional_world_v2")),
            "seed": seed,
            "output_dir": str(root),
        }
        runtime_config["model"] = {**dict(model), "seed": seed}
        rows = run_conditions(evaluated_records, build_vlm(runtime_config), conditions)
        lookup = {record.scene_id: record for record in evaluated_records}
        for row in rows:
            record = lookup[str(row["scene_id"])]
            row["question_type"] = record.metadata["question_type"]
            row["hop_depth"] = record.metadata["hop_depth"]
            row["primary_relation"] = record.metadata["primary_relation"]
            row["benchmark_seed"] = seed
            all_predictions.append(row)
        per_seed.append(
            {
                "seed": seed,
                "generated_count": len(records),
                "evaluated_count": len(evaluated_records),
                "conditions": _condition_accuracy(rows),
                "slices": _slice_accuracy(rows),
                "integrity": integrity,
            }
        )

    aggregate: dict[str, Any] = {}
    for condition in conditions:
        aggregate[condition] = seed_summary(
            [float(row["conditions"][condition]["accuracy"]) for row in per_seed],
            float(evaluation.get("confidence_level", 0.95)),
        )
    metrics = {
        "schema_version": 1,
        "benchmark": str(benchmark.get("name", "synthetic_compositional_world_v2")),
        "dataset_version": "generator-v2",
        "model_backend": str(model.get("backend", "reference")),
        "split_strategy": split_strategy,
        "generated_sample_count_per_seed": count,
        "evaluation_splits": list(evaluation_splits) if evaluation_splits else ["all"],
        "seed_count": len(seeds),
        "seeds": seeds,
        "conditions": aggregate,
        "per_seed": per_seed,
        "distribution": {key: dict(values) for key, values in distribution.items()},
        "integrity": {
            "valid": not integrity_failures,
            "failures": integrity_failures,
            "caption_graph_equivalence_checked": True,
            "answers_recomputed_from_graph": True,
            "split_leakage_checked": True,
            "image_reproducibility_checked": bool(benchmark.get("write_images", True)),
        },
    }
    with (root / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in all_predictions:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    reference_backend = str(model.get("backend", "reference")) == "reference"
    registry_compatibility = _registry_compatibility(config)
    metrics["registry_compatibility"] = registry_compatibility
    _write_json(root / "metrics.json", metrics)
    decision = {
        "schema_version": 1,
        "benchmark": metrics["benchmark"],
        "instrument_validation": "PASS" if metrics["integrity"]["valid"] else "FAIL",
        "scientific_decision": "NOT_APPLICABLE" if reference_backend else "INCONCLUSIVE",
        "reason": (
            "Reference backend validates infrastructure only; it cannot support a model claim."
            if reference_backend
            else "Model results require the preregistered EXP001/EXP002/EXP003 decision engine."
        ),
        "seed_count": len(seeds),
        "required_seed_count": required_seeds,
        "critical_experiment": critical,
        "artifacts": {
            "metrics": "metrics.json",
            "predictions": "predictions.jsonl",
        },
        "experiment_registry_compatibility": registry_compatibility,
        "next_action": "Run the frozen target VLM through a preregistered Phase-1 protocol.",
    }
    (root / "decision_report.yaml").write_text(
        yaml.safe_dump(decision, sort_keys=False), encoding="utf-8", newline="\n"
    )
    return metrics


def generate_example_dataset(
    config_path: str | Path,
    *,
    count: int = 1000,
    output_dir: str | Path = "outputs/synthetic_world_example",
    seed: int | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    benchmark = dict(config.get("benchmark", {}))
    actual_seed = int(benchmark.get("seed", 20260819) if seed is None else seed)
    generator = SyntheticWorldGenerator(
        GeneratorConfig(
            seed=actual_seed,
            image_size=int(benchmark.get("resolution", 192)),
            write_images=bool(benchmark.get("write_images", True)),
            style=str(benchmark.get("style", "flat")),
            min_hops=int(benchmark.get("min_hops", 1)),
            max_hops=int(benchmark.get("max_hops", 4)),
        )
    )
    strategy = str(benchmark.get("split_strategy", "composition"))
    records = generator.generate(
        count,
        output_dir=output_dir,
        seed=actual_seed,
        split_strategy=strategy,
    )
    report = validate_dataset(
        records,
        split_strategy=strategy,
        check_images=bool(benchmark.get("write_images", True)),
    )
    _write_json(Path(output_dir) / "validation_report.json", report)
    return report


def _condition_accuracy(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(bool(row["correct"]))
    return {
        condition: {"accuracy": sum(values) / len(values), "n": len(values)}
        for condition, values in sorted(grouped.items())
    }


def _slice_accuracy(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[f"question_type={row['question_type']}"] .append(bool(row["correct"]))
        grouped[f"hop_depth={row['hop_depth']}"] .append(bool(row["correct"]))
    return {
        key: {"accuracy": sum(values) / len(values), "n": len(values)}
        for key, values in sorted(grouped.items())
    }


def _t_critical_95(degrees_of_freedom: int) -> float:
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        15: 2.131,
        20: 2.086,
        30: 2.042,
    }
    if degrees_of_freedom in table:
        return table[degrees_of_freedom]
    if degrees_of_freedom < 1:
        return 0.0
    smaller = max(key for key in table if key <= min(degrees_of_freedom, 30))
    return table[smaller] if degrees_of_freedom <= 30 else 1.96


def _registry_compatibility(config: dict[str, Any]) -> dict[str, Any]:
    registry = load_experiment_registry()
    requested = config.get("registry", {}).get(
        "compatible_experiments", ("EXP001", "EXP002", "EXP003")
    )
    registrations = {
        str(item["experiment_id"]): item for item in registry.get("experiments", [])
    }
    supported_conditions = {
        "image_only",
        "object_list",
        "caption",
        "scene_graph",
        "partial_graph",
        "random_graph",
        "corrupted_graph",
    }
    report: dict[str, Any] = {}
    for experiment_id in requested:
        registration = registrations.get(str(experiment_id))
        if registration is None:
            raise ValueError(f"unknown compatible experiment in registry: {experiment_id}")
        inputs = [str(value) for value in registration.get("input_conditions", [])]
        report[str(experiment_id)] = {
            "api_compatible": set(inputs) <= supported_conditions,
            "input_conditions": inputs,
            "registered_dataset_version": registration.get("dataset", {}).get("version"),
            "v2_registration_required": (
                registration.get("dataset", {}).get("version") != "generator-v2"
            ),
        }
    return report


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


__all__ = ["evaluate_synthetic", "generate_example_dataset", "seed_summary"]
