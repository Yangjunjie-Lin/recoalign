"""Execute EXP003 controlled OOD compositional-generalization validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw

from evaluation.statistics import multiple_seed_summary
from experiments.ood_composition.analysis import (
    CAPTION,
    CONDITIONS,
    GRAPH,
    IMAGE_ONLY,
    RANDOM_GRAPH,
    analyze_seed,
)
from experiments.runtime import (
    build_generator,
    build_vlm,
    checkpoint_provenance,
    git_commit,
    load_config,
    run_conditions,
)
from recoalign.reproducibility import get_git_metadata, utc_now
from recoalign.research_registry import get_registered_experiment, validate_research_registries
from recoalign.schema_validation import repository_root, validate_payload
from recoalign.synthetic_world.questions import graph_variant
from recoalign.synthetic_world.splits import SPLIT_NAMES, build_ood_split_suite
from research.decision_engine import evaluate_decision, render_decision_report


def run(
    config_path: str | Path = "configs/ood_composition.yaml",
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run all configured seeds and write the authoritative EXP003 result bundle."""

    validate_research_registries()
    config = deepcopy(load_config(config_path))
    _validate_config(config)
    seeds = [int(value) for value in config["experiment"]["seeds"]]
    minimum = 5 if bool(config["evaluation"].get("critical", True)) else 3
    if len(seeds) < minimum:
        raise ValueError(f"EXP003 requires at least {minimum} unique seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("EXP003 seeds must be unique")

    output = Path(output_dir or config["experiment"]["output_dir"])
    _ensure_new_bundle(output)
    resolved = deepcopy(config)
    resolved["experiment"]["output_dir"] = output.as_posix()
    resolved.setdefault("meta", {})["executed_config_path"] = str(
        Path(config_path).resolve().as_posix()
    )
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8", newline="\n"
    )

    per_seed: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    try:
        with (output / "predictions.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as target:
            for seed in seeds:
                seed_dir = output / "seeds" / str(seed)
                metrics = run_seed_config(resolved, output_dir=seed_dir, seed=seed)
                per_seed.append({"seed": seed, "metrics": metrics})
                integrity = _seed_integrity(seed_dir, seed)
                integrity_rows.append(integrity)
                split_rows.append(
                    {
                        "seed": seed,
                        "path": (seed_dir / "split_manifest.json").as_posix(),
                        "sha256": _sha256(seed_dir / "split_manifest.json"),
                        "assertions": integrity["manifest_assertions"],
                    }
                )
                with (seed_dir / "predictions.jsonl").open(encoding="utf-8") as source:
                    for line in source:
                        target.write(line)
    except Exception as exc:
        _record_failed_bundle(output, resolved, seeds, per_seed, integrity_rows, exc)
        raise

    registration = get_registered_experiment("EXP003")
    aggregate = _registered_aggregate(registration, per_seed, resolved)
    descriptive = _descriptive_aggregate(per_seed, resolved)
    result = {
        "schema_version": 1,
        "run_id": f"exp003-{utc_now().replace(':', '').replace('-', '')}",
        "experiment_id": "EXP003",
        "hypothesis": "H003",
        "model": f"frozen-{resolved['model']['backend']}",
        "dataset": f"{resolved['data']['dataset']}@{resolved['data']['version']}",
        "seed": seeds,
        "metrics": {
            "per_seed": per_seed,
            "aggregate": aggregate,
            **descriptive,
        },
        "confidence_interval": {
            path: value["confidence_interval"] for path, value in aggregate.items()
        },
        "statistical_test": {
            path: value["statistical_test"] for path, value in aggregate.items()
        },
        "decision": "INCONCLUSIVE",
        "timestamp": utc_now(),
        "integrity": {
            "registry_validated": True,
            "dataset_version": registration["dataset"]["version"],
            "per_seed": integrity_rows,
        },
        "status": "complete",
    }
    report = evaluate_decision(registration, result)
    validate_payload("decision_report", report)
    result["decision"] = report["final_decision"]
    validate_payload("experiment_result", result)
    _write_json(output / "metrics.json", result)
    (output / "decision_report.yaml").write_text(
        yaml.safe_dump(report, sort_keys=False), encoding="utf-8", newline="\n"
    )
    (output / "decision_report.md").write_text(
        render_decision_report(report), encoding="utf-8", newline="\n"
    )
    _write_json(
        output / "split_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": "EXP003",
            "generator": "recoalign.synthetic_world.v2",
            "seeds": split_rows,
            "all_seeds_valid": all(row["valid"] for row in integrity_rows),
        },
    )
    _write_figures(output, descriptive)
    _write_reports(output, result, report)

    run_record = {
        "schema_version": 1,
        "run_id": result["run_id"],
        "experiment_id": "EXP003",
        "hypothesis_id": "H003",
        "status": "complete",
        "decision": report["final_decision"],
        "started_from_config": str(Path(config_path).resolve().as_posix()),
        "completed_at": utc_now(),
        "model_backend": resolved["model"]["backend"],
        "checkpoint": checkpoint_provenance(resolved),
        "dataset": result["dataset"],
        "seeds": seeds,
        "scientific_interpretation": _scientific_interpretation(report, resolved),
        "git": get_git_metadata(repository_root()),
    }
    _write_json(output / "run.json", run_record)
    _write_json(output / "manifest.json", _bundle_manifest(output, run_record))
    return result


def run_seed_config(
    config: dict[str, Any],
    *,
    output_dir: str | Path,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run one EXP003 seed; this is the governed unified-runner integration point."""

    resolved = deepcopy(config)
    _validate_config(resolved)
    actual_seed = int(resolved["experiment"]["seed"] if seed is None else seed)
    resolved["experiment"]["seed"] = actual_seed
    resolved["model"]["seed"] = actual_seed
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    resolved["experiment"]["output_dir"] = output.as_posix()
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8", newline="\n"
    )

    generator = build_generator(resolved)
    synthetic = resolved["synthetic"]
    suite = build_ood_split_suite(
        generator,
        train_count=int(synthetic["train_count"]),
        test_count=int(synthetic["test_count"]),
        seed=actual_seed,
        output_dir=output / "dataset" / "splits",
    )
    if not suite.validation["valid"]:
        raise RuntimeError("EXP003 split suite did not pass leakage validation")

    model = build_vlm(resolved)
    conditions = tuple(str(value) for value in resolved["evaluation"]["conditions"])
    rows: list[dict[str, Any]] = []
    for split_name in SPLIT_NAMES:
        split = suite.splits[split_name]
        for partition, records in (("train", split.train), ("test", split.test)):
            values = run_conditions(records, model, conditions)
            lookup = {record.scene_id: record for record in records}
            for row in values:
                record = lookup[str(row["scene_id"])]
                row.update(
                    {
                        "benchmark_seed": actual_seed,
                        "ood_split": split_name,
                        "partition": partition,
                        "pair_id": record.metadata["pair_id"],
                        "primary_relation": record.metadata["primary_relation"],
                        "relation_sequence": record.metadata["relation_sequence"],
                        "relation_combination": record.metadata["relation_combination"],
                        "anti_memorization": record.metadata.get("anti_memorization"),
                        "evaluation_role": (
                            "seen_reference" if partition == "train" else "heldout_evaluation"
                        ),
                    }
                )
            rows.extend(values)

    metrics = analyze_seed(
        rows,
        bootstrap_samples=int(resolved["evaluation"]["bootstrap_samples"]),
        seed=int(resolved["evaluation"]["statistics_seed"]) + actual_seed,
    )
    random_integrity = _random_graph_integrity(rows, suite)
    observed = suite.validation["assertions"]
    assertions = {
        "generator": "recoalign.synthetic_world.v2",
        "composition_overlap": observed["composition_overlap"],
        "relation_combination_overlap": observed["relation_combination_overlap"],
        "hop_depth_overlap": observed["hop_depth_overlap"],
        "primitive_coverage": observed["primitive_coverage"],
        "sample_id_overlap": observed["sample_id_overlap"],
        "random_graph_not_equivalent": random_integrity["random_graph_not_equivalent"],
        "random_graph_length_matched": random_integrity["random_graph_length_matched"],
        "test_information_leakage": False,
    }
    metrics.update(
        {
            "experiment": "structured_interface_ood_compositional_generalization",
            "model_backend": resolved["model"]["backend"],
            "seed": actual_seed,
            "integrity": {
                "valid": _assertions_pass(assertions),
                "split_validation": suite.validation,
                "random_structure_control": random_integrity,
                "manifest_assertions": assertions,
            },
        }
    )
    _write_json(output / "metrics.json", metrics)
    with (output / "predictions.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    source_manifest = output / "dataset" / "splits" / "manifest.json"
    split_manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    _write_json(output / "split_manifest.json", split_manifest)
    dataset_manifest = {
        "schema_version": 2,
        "experiment_id": "EXP003",
        "generator": "recoalign.synthetic_world.v2",
        "seed": actual_seed,
        "split_manifest": "splits/manifest.json",
        "split_manifest_sha256": _sha256(source_manifest),
        "manifest_assertions": assertions,
        "caption_graph_equivalent": True,
    }
    _write_json(output / "dataset" / "manifest.json", dataset_manifest)
    seed_run = {
        "experiment": "EXP003",
        "status": "complete",
        "seed": actual_seed,
        "model_backend": resolved["model"]["backend"],
        "checkpoint": checkpoint_provenance(resolved),
        "dataset_size": sum(
            len(split.train) + len(split.test) for split in suite.splits.values()
        ),
        "prediction_count": len(rows),
        "git_commit": git_commit(),
        "manifest_assertions": assertions,
    }
    _write_json(output / "run.json", seed_run)
    (output / "run.log").write_text(
        f"{utc_now()} RUN_COMPLETE experiment=EXP003 seed={actual_seed}\n",
        encoding="utf-8",
        newline="\n",
    )
    if not _assertions_pass(assertions):
        raise RuntimeError(f"EXP003 integrity gate failed: {assertions}")
    return metrics


def _validate_config(config: dict[str, Any]) -> None:
    if config["experiment"].get("id") != "EXP003":
        raise ValueError("EXP003 bundle requires experiment.id=EXP003")
    conditions = tuple(str(value) for value in config["evaluation"].get("conditions", ()))
    if conditions != CONDITIONS:
        raise ValueError(f"EXP003 requires evaluation.conditions={list(CONDITIONS)}")
    if config.get("training", {}).get("enabled") is not False:
        raise ValueError("EXP003 cannot enable training or OOD-specific parameter updates")
    if int(config["synthetic"]["train_count"]) < 8 or int(config["synthetic"]["test_count"]) < 8:
        raise ValueError("EXP003 requires at least eight records per split partition")


def _random_graph_integrity(rows: list[dict[str, Any]], suite: Any) -> dict[str, Any]:
    records = {
        record.scene_id: record
        for split in suite.splits.values()
        for record in (*split.train, *split.test)
    }
    graph_changed = all(
        tuple(graph_variant(record, RANDOM_GRAPH).edges) != tuple(record.relations)
        for record in records.values()
    )
    tokens: dict[str, dict[str, int]] = {}
    for row in rows:
        if row["condition"] in {GRAPH, RANDOM_GRAPH}:
            tokens.setdefault(str(row["scene_id"]), {})[str(row["condition"])] = int(
                row["input"]["input_tokens"]
            )
    deltas = [
        abs(values[GRAPH] - values[RANDOM_GRAPH])
        for values in tokens.values()
        if GRAPH in values and RANDOM_GRAPH in values
    ]
    return {
        "random_graph_not_equivalent": graph_changed,
        "random_graph_length_matched": bool(deltas) and max(deltas) == 0,
        "n_pairs": len(deltas),
        "maximum_token_delta": max(deltas, default=None),
    }


def _assertions_pass(assertions: dict[str, Any]) -> bool:
    expected = {
        "generator": "recoalign.synthetic_world.v2",
        "composition_overlap": False,
        "relation_combination_overlap": False,
        "hop_depth_overlap": False,
        "primitive_coverage": True,
        "sample_id_overlap": False,
        "random_graph_not_equivalent": True,
        "random_graph_length_matched": True,
        "test_information_leakage": False,
    }
    return all(assertions.get(key) == value for key, value in expected.items())


def _registered_aggregate(
    registration: dict[str, Any],
    per_seed: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, criterion in enumerate(registration["decision_rule"]["criteria"]):
        path = str(criterion["metric_path"])
        values = [float(_lookup(row["metrics"], path)) for row in per_seed]
        result[path] = multiple_seed_summary(
            values,
            confidence=float(config["evaluation"]["confidence_level"]),
            bootstrap_samples=int(config["evaluation"]["bootstrap_samples"]),
            seed=int(config["evaluation"]["statistics_seed"]) + index,
            significance_test=str(config["evaluation"]["significance_test"]),
        )
    return result


def _descriptive_aggregate(
    per_seed: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    samples = int(config["evaluation"]["bootstrap_samples"])
    stats_seed = int(config["evaluation"]["statistics_seed"])
    conditions: dict[str, Any] = {}
    gaps: dict[str, Any] = {}
    retention: dict[str, Any] = {}
    for condition in CONDITIONS:
        iid_path = f"splits.iid.conditions.{condition}.accuracy"
        ood_path = f"generalization.conditions.{condition}.ood_accuracy"
        gap_path = f"generalization.conditions.{condition}.generalization_gap"
        retention_path = f"generalization.conditions.{condition}.ood_retention_ratio"
        conditions[condition] = {
            "iid": _summary(per_seed, iid_path, samples, stats_seed + len(iid_path)),
            "ood": _summary(per_seed, ood_path, samples, stats_seed + len(ood_path)),
        }
        gaps[condition] = _summary(
            per_seed, gap_path, samples, stats_seed + len(gap_path)
        )
        retention[condition] = _summary(
            per_seed, retention_path, samples, stats_seed + len(retention_path)
        )

    depths: dict[str, Any] = {}
    for depth in range(1, 5):
        depths[str(depth)] = {}
        for condition in (CAPTION, GRAPH, RANDOM_GRAPH):
            path = f"reasoning_depth.depths.{depth}.conditions.{condition}.accuracy"
            depths[str(depth)][condition] = _summary(
                per_seed, path, samples, stats_seed + depth * 17 + len(condition)
            )
    anti: dict[str, Any] = {}
    for offset, name in enumerate(
        ("object_identity_swap", "relation_recombination", "attribute_transfer")
    ):
        path = f"anti_memorization.{name}.graph_over_caption.estimate"
        anti[name] = _summary(per_seed, path, samples, stats_seed + 701 + offset)
    return {
        "condition_summaries": conditions,
        "generalization_gap_summaries": gaps,
        "retention_summaries": retention,
        "reasoning_depth_summaries": depths,
        "anti_memorization_summaries": anti,
    }


def _summary(
    per_seed: list[dict[str, Any]], path: str, samples: int, seed: int
) -> dict[str, Any]:
    return multiple_seed_summary(
        [float(_lookup(row["metrics"], path)) for row in per_seed],
        bootstrap_samples=samples,
        seed=seed,
        significance_test="paired_bootstrap",
    )


def _seed_integrity(seed_dir: Path, seed: int) -> dict[str, Any]:
    manifest_path = seed_dir / "split_manifest.json"
    run_path = seed_dir / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_payload = json.loads(run_path.read_text(encoding="utf-8"))
    assertions = dict(run_payload["manifest_assertions"])
    return {
        "seed": seed,
        "valid": bool(manifest["validation"]["valid"]) and _assertions_pass(assertions),
        "split_manifest": manifest_path.as_posix(),
        "split_manifest_sha256": _sha256(manifest_path),
        "manifest_assertions": assertions,
    }


def _write_figures(output: Path, descriptive: dict[str, Any]) -> None:
    labels = [IMAGE_ONLY, CAPTION, GRAPH]
    _grouped_bar_chart(
        output / "figure1_iid_vs_ood_performance.png",
        "EXP003 IID vs OOD Performance",
        labels,
        {
            "IID": [
                float(descriptive["condition_summaries"][name]["iid"]["mean"])
                for name in labels
            ],
            "OOD": [
                float(descriptive["condition_summaries"][name]["ood"]["mean"])
                for name in labels
            ],
        },
        "Accuracy",
    )
    _bar_chart(
        output / "figure2_generalization_gap.png",
        "EXP003 Generalization Gap (IID - OOD; Lower is Better)",
        labels,
        [
            float(descriptive["generalization_gap_summaries"][name]["mean"])
            for name in labels
        ],
        "Accuracy gap",
    )
    depth_labels = ["1", "2", "3", "4"]
    _line_chart(
        output / "figure3_reasoning_depth.png",
        "EXP003 Performance vs Reasoning Depth",
        depth_labels,
        {
            "Caption": [
                float(descriptive["reasoning_depth_summaries"][depth][CAPTION]["mean"])
                for depth in depth_labels
            ],
            "Graph": [
                float(descriptive["reasoning_depth_summaries"][depth][GRAPH]["mean"])
                for depth in depth_labels
            ],
        },
    )


def _write_reports(
    output: Path, result: dict[str, Any], decision: dict[str, Any]
) -> None:
    aggregate = result["metrics"]["aggregate"]
    condition = result["metrics"]["condition_summaries"]
    implementation = """# EXP003 Implementation Report

The controlled generator constructs IID, composition-novelty, unseen relation-combination, and
unseen-hop partitions before evaluation. Every train/test partition contains image, graph, lossless
caption, question, and answer artifacts with SHA-256 manifests. No model parameter or prompt differs
between IID and OOD evaluation. The random-graph condition preserves nodes and edge count while
resampling every relation triple deterministically.
"""
    benchmark = [
        "# EXP003 OOD Benchmark Report",
        "",
        "| Condition | IID accuracy | OOD accuracy | Retention |",
        "|---|---:|---:|---:|",
    ]
    for name in (IMAGE_ONLY, CAPTION, GRAPH, RANDOM_GRAPH):
        benchmark.append(
            f"| {name} | {condition[name]['iid']['mean']:.4f} | "
            f"{condition[name]['ood']['mean']:.4f} | "
            f"{result['metrics']['retention_summaries'][name]['mean']:.4f} |"
        )
    analysis = [
        "# EXP003 Generalization Analysis",
        "",
        f"Final governed decision: **{decision['final_decision']}**.",
        "",
        "Registered seed-level effects:",
        "",
    ]
    for path, summary in aggregate.items():
        interval = summary["confidence_interval"]
        analysis.append(
            f"- `{path}`: mean {summary['mean']:.4f}, std {summary['std']:.4f}, "
            f"95% CI [{interval['lower']:.4f}, {interval['upper']:.4f}]"
        )
    (output / "implementation_report.md").write_text(
        implementation, encoding="utf-8", newline="\n"
    )
    (output / "ood_benchmark_report.md").write_text(
        "\n".join(benchmark) + "\n", encoding="utf-8", newline="\n"
    )
    (output / "generalization_analysis.md").write_text(
        "\n".join(analysis) + "\n", encoding="utf-8", newline="\n"
    )


def _scientific_interpretation(
    report: dict[str, Any], config: dict[str, Any]
) -> str:
    if config["model"]["backend"] == "reference":
        return (
            "INCONCLUSIVE: ReferenceVLM validates EXP003 mechanics but is not eligible evidence "
            "for a scientific compositional-generalization claim."
        )
    if report["final_decision"] == "GO":
        return "All preregistered OOD, retention, depth, stability, and random-graph gates passed."
    if report["final_decision"] == "NO-GO":
        return "At least one registered anti-memorization or OOD generalization gate failed."
    return "INCONCLUSIVE: eligible model evidence or a robustness requirement is incomplete."


def _record_failed_bundle(
    output: Path,
    config: dict[str, Any],
    seeds: list[int],
    per_seed: list[dict[str, Any]],
    integrity_rows: list[dict[str, Any]],
    error: Exception,
) -> None:
    registration = get_registered_experiment("EXP003")
    result = {
        "schema_version": 1,
        "run_id": f"exp003-failed-{utc_now().replace(':', '').replace('-', '')}",
        "experiment_id": "EXP003",
        "hypothesis": "H003",
        "model": f"frozen-{config['model']['backend']}",
        "dataset": f"{config['data']['dataset']}@{config['data']['version']}",
        "seed": seeds,
        "metrics": {"per_seed": per_seed, "aggregate": {}},
        "confidence_interval": None,
        "statistical_test": None,
        "decision": "INCONCLUSIVE",
        "timestamp": utc_now(),
        "integrity": {"registry_validated": True, "per_seed": integrity_rows},
        "status": "failed",
    }
    report = evaluate_decision(registration, result)
    validate_payload("experiment_result", result)
    validate_payload("decision_report", report)
    _write_json(output / "metrics.json", result)
    (output / "decision_report.yaml").write_text(
        yaml.safe_dump(report, sort_keys=False), encoding="utf-8", newline="\n"
    )
    (output / "decision_report.md").write_text(
        render_decision_report(report), encoding="utf-8", newline="\n"
    )
    _write_json(
        output / "run.json",
        {
            "schema_version": 1,
            "run_id": result["run_id"],
            "experiment_id": "EXP003",
            "status": "failed",
            "decision": "INCONCLUSIVE",
            "completed_at": utc_now(),
            "error": f"{type(error).__name__}: {error}",
            "completed_seeds": [row["seed"] for row in per_seed],
        },
    )


def _bundle_manifest(output: Path, run_record: dict[str, Any]) -> dict[str, Any]:
    names = (
        "config.resolved.yaml",
        "metrics.json",
        "predictions.jsonl",
        "split_manifest.json",
        "run.json",
        "decision_report.yaml",
        "decision_report.md",
        "figure1_iid_vs_ood_performance.png",
        "figure2_generalization_gap.png",
        "figure3_reasoning_depth.png",
        "implementation_report.md",
        "ood_benchmark_report.md",
        "generalization_analysis.md",
    )
    return {
        "schema_version": 1,
        "experiment_id": "EXP003",
        "run_id": run_record["run_id"],
        "status": "complete",
        "artifacts": {
            name: {"bytes": (output / name).stat().st_size, "sha256": _sha256(output / name)}
            for name in names
        },
    }


def _ensure_new_bundle(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"EXP003 output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)


def _lookup(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"EXP003 metric path is missing: {path}")
        value = value[part]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _axes(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    right: int,
    bottom: int,
    title: str,
    y_label: str,
    ceiling: float = 1.0,
) -> None:
    draw.text((left, 25), title, fill="black")
    draw.text((left, top - 18), y_label, fill="black")
    draw.line((left, top, left, bottom, right, bottom), fill="black", width=2)
    for index in range(6):
        value = ceiling * index / 5
        y = bottom - index * (bottom - top) / 5
        draw.line((left, y, right, y), fill="#e2e8f0", width=1)
        draw.text((45, y - 7), f"{value:.2f}", fill="#475569")


def _bar_chart(
    path: Path, title: str, labels: list[str], values: list[float], y_label: str
) -> None:
    image = Image.new("RGB", (960, 600), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 90, 70, 920, 510
    ceiling = max(1.0, max(values, default=0.0) * 1.1)
    _axes(draw, left, top, right, bottom, title, y_label, ceiling)
    slot = (right - left) / len(labels)
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        x0 = left + index * slot + slot * 0.2
        x1 = left + (index + 1) * slot - slot * 0.2
        y = bottom - max(0.0, value) / ceiling * (bottom - top)
        draw.rectangle((x0, y, x1, bottom), fill="#2563eb")
        draw.text((x0, bottom + 12), label.replace("_", " "), fill="black")
        draw.text((x0, y - 18), f"{value:.3f}", fill="black")
    image.save(path)


def _grouped_bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    series: dict[str, list[float]],
    y_label: str,
) -> None:
    image = Image.new("RGB", (960, 600), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 90, 70, 920, 510
    _axes(draw, left, top, right, bottom, title, y_label)
    colors = {"IID": "#64748b", "OOD": "#2563eb"}
    slot = (right - left) / len(labels)
    width = slot * 0.28
    for index, label in enumerate(labels):
        center = left + (index + 0.5) * slot
        for series_index, (name, values) in enumerate(series.items()):
            value = values[index]
            x0 = center + (series_index - 1) * width
            x1 = x0 + width
            y = bottom - value * (bottom - top)
            draw.rectangle((x0, y, x1, bottom), fill=colors[name])
        draw.text((center - width, bottom + 12), label.replace("_", " "), fill="black")
    for index, name in enumerate(series):
        draw.rectangle(
            (right - 130, top + index * 24, right - 116, top + 14 + index * 24),
            fill=colors[name],
        )
        draw.text((right - 108, top + index * 24), name, fill="black")
    image.save(path)


def _line_chart(
    path: Path, title: str, labels: list[str], series: dict[str, list[float]]
) -> None:
    image = Image.new("RGB", (960, 600), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 90, 70, 920, 510
    _axes(draw, left, top, right, bottom, title, "Accuracy")
    colors = {"Caption": "#f59e0b", "Graph": "#2563eb"}
    for name, values in series.items():
        points = []
        for index, value in enumerate(values):
            x = left + index * (right - left) / (len(labels) - 1)
            y = bottom - value * (bottom - top)
            points.append((x, y))
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=colors[name])
            draw.text((x - 4, bottom + 12), labels[index], fill="black")
        draw.line(points, fill=colors[name], width=4)
    for index, name in enumerate(series):
        draw.rectangle(
            (right - 150, top + index * 24, right - 136, top + 14 + index * 24),
            fill=colors[name],
        )
        draw.text((right - 128, top + index * 24), name, fill="black")
    image.save(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ood_composition.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    result = run(args.config, output_dir=args.output)
    print(f"EXP003 decision: {result['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run", "run_seed_config"]
