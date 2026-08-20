"""Execute EXP001 with information controls, ablations, statistics, and figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw

from evaluation.metrics import write_metrics
from evaluation.statistics import multiple_seed_summary
from experiments.graph_vs_text.analysis import PRIMARY_CONDITIONS, SETTINGS, analyze_seed
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
from recoalign.synthetic_world.validation import validate_dataset
from research.decision_engine import evaluate_decision, render_decision_report
from synthetic_world.compositional_tasks.tasks import ABLATION_CONDITIONS


def run(
    config_path: str | Path = "configs/graph_vs_text.yaml",
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run all configured seeds and write the authoritative EXP001 bundle."""

    validate_research_registries()
    config = deepcopy(load_config(config_path))
    if config["experiment"].get("id") != "EXP001":
        raise ValueError("EXP001 bundle requires experiment.id=EXP001")
    configured_settings = tuple(str(value) for value in config["evaluation"].get("settings", ()))
    if configured_settings != SETTINGS:
        raise ValueError("EXP001 requires settings [natural, token_matched] in that order")
    seeds = [int(value) for value in config["experiment"]["seeds"]]
    minimum = 5 if bool(config["evaluation"].get("critical", True)) else 3
    if len(seeds) < minimum:
        raise ValueError(f"EXP001 requires at least {minimum} unique seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("EXP001 seeds must be unique")
    output = Path(output_dir or config["experiment"]["output_dir"])
    _ensure_new_bundle(output)
    resolved = deepcopy(config)
    resolved["experiment"]["output_dir"] = output.as_posix()
    resolved["meta"]["executed_config_path"] = str(Path(config_path).resolve().as_posix())
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8", newline="\n"
    )

    per_seed: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    try:
        with (output / "predictions.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as target:
            for seed in seeds:
                seed_dir = output / "seeds" / str(seed)
                metrics = run_seed_config(resolved, output_dir=seed_dir, seed=seed)
                per_seed.append({"seed": seed, "metrics": metrics})
                integrity_rows.append(_seed_integrity(seed_dir, seed))
                with (seed_dir / "predictions.jsonl").open(encoding="utf-8") as source:
                    for line in source:
                        target.write(line)
    except Exception as exc:
        _record_failed_bundle(output, resolved, seeds, per_seed, integrity_rows, exc)
        raise

    registration = get_registered_experiment("EXP001")
    aggregate = _registered_aggregate(registration, per_seed, resolved)
    descriptive = _descriptive_aggregate(per_seed, resolved)
    result = {
        "schema_version": 1,
        "run_id": f"exp001-{utc_now().replace(':', '').replace('-', '')}",
        "experiment_id": "EXP001",
        "hypothesis": "H001",
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
    _write_figures(output, descriptive)

    run_record = {
        "schema_version": 1,
        "run_id": result["run_id"],
        "experiment_id": "EXP001",
        "hypothesis_id": "H001",
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
    """Run one seed; this is also the governed unified-runner integration point."""

    resolved = deepcopy(config)
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
    records = generator.generate(
        int(synthetic.get("count", 120)),
        output_dir=output / "dataset",
        seed=actual_seed,
        split="test",
    )
    integrity = validate_dataset(
        records, check_images=bool(synthetic.get("write_images", True))
    )
    model = build_vlm(resolved)
    tolerance = int(resolved["evaluation"].get("token_match_tolerance", 1))
    rows: list[dict[str, Any]] = []
    for setting in tuple(str(value) for value in resolved["evaluation"].get("settings", SETTINGS)):
        values = run_conditions(
            records,
            model,
            PRIMARY_CONDITIONS,
            setting=setting,
            token_match_tolerance=tolerance,
        )
        for row in values:
            row["scope"] = "iid_primary"
        rows.extend(values)

    ablations = tuple(
        str(value)
        for value in resolved["evaluation"].get("ablations", ABLATION_CONDITIONS)
    )
    ablation_rows = run_conditions(
        records,
        model,
        ablations,
        setting="natural",
        token_match_tolerance=tolerance,
    )
    for row in ablation_rows:
        row["scope"] = "ablation"
    rows.extend(ablation_rows)

    ood_rows: list[dict[str, Any]] = []
    ood = dict(resolved.get("ood", {}))
    if bool(ood.get("enabled", True)):
        _train, test = generator.generate_ood_splits(
            int(ood.get("train_count", 24)),
            int(ood.get("test_count", 24)),
            output_dir=output / "ood_dataset",
            seed=actual_seed,
        )
        validate_dataset(test, check_images=bool(synthetic.get("write_images", True)))
        for setting in tuple(
            str(value) for value in resolved["evaluation"].get("settings", SETTINGS)
        ):
            values = run_conditions(
                test,
                model,
                ("caption", "scene_graph"),
                setting=setting,
                token_match_tolerance=tolerance,
            )
            for row in values:
                row["scope"] = "ood_preliminary"
            ood_rows.extend(values)

    all_rows = [*rows, *ood_rows]
    metrics = analyze_seed(
        rows,
        ood_rows,
        bootstrap_samples=int(resolved["evaluation"]["bootstrap_samples"]),
        statistics_seed=int(resolved["evaluation"]["statistics_seed"]),
    )
    metrics.update(
        {
            "model_backend": resolved["model"]["backend"],
            "seed": actual_seed,
            "integrity": integrity,
        }
    )
    write_metrics(output / "metrics.json", metrics)
    with (output / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    seed_run = {
        "experiment": "EXP001",
        "status": "complete",
        "seed": actual_seed,
        "model_backend": resolved["model"]["backend"],
        "checkpoint": checkpoint_provenance(resolved),
        "dataset_size": len(records),
        "prediction_count": len(all_rows),
        "git_commit": git_commit(),
        "information_control_passed": metrics["information_control"]["passed"],
    }
    _write_json(output / "run.json", seed_run)
    _write_json(
        output / "manifest.json",
        {
            **seed_run,
            "dataset_manifest": "dataset/manifest.json",
            "ood_manifest": "ood_dataset/manifest.json" if ood_rows else None,
        },
    )
    (output / "run.log").write_text(
        f"{utc_now()} RUN_COMPLETE experiment=EXP001 seed={actual_seed}\n",
        encoding="utf-8",
        newline="\n",
    )
    return metrics


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
        )
    return result


def _descriptive_aggregate(
    per_seed: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    samples = int(config["evaluation"]["bootstrap_samples"])
    stats_seed = int(config["evaluation"]["statistics_seed"])
    conditions: dict[str, Any] = {}
    for setting in SETTINGS:
        conditions[setting] = {}
        for condition in PRIMARY_CONDITIONS:
            path = (
                f"conditions.{condition}.accuracy"
                if setting == "natural"
                else f"token_accounting.{setting}:{condition}.accuracy"
            )
            values = [float(_lookup(row["metrics"], path)) for row in per_seed]
            conditions[setting][condition] = multiple_seed_summary(
                values, bootstrap_samples=samples, seed=stats_seed + len(path)
            )
    depths: dict[str, Any] = {}
    for setting in SETTINGS:
        depths[setting] = {}
        for depth in range(1, 5):
            depths[setting][str(depth)] = {}
            for condition in ("caption", "scene_graph"):
                path = f"reasoning_depth.{setting}.{depth}.conditions.{condition}.accuracy"
                values = [float(_lookup(row["metrics"], path)) for row in per_seed]
                depths[setting][str(depth)][condition] = multiple_seed_summary(
                    values, bootstrap_samples=samples, seed=stats_seed + depth + len(condition)
                )
    token_efficiency: dict[str, Any] = {}
    for condition in PRIMARY_CONDITIONS:
        path = f"token_accounting.natural:{condition}.accuracy_per_1000_input_tokens"
        values = [float(_lookup(row["metrics"], path)) for row in per_seed]
        token_efficiency[condition] = multiple_seed_summary(
            values, bootstrap_samples=samples, seed=stats_seed + len(condition) * 11
        )
    return {
        "condition_summaries": conditions,
        "reasoning_depth_summaries": depths,
        "token_efficiency_summaries": token_efficiency,
    }


def _seed_integrity(seed_dir: Path, seed: int) -> dict[str, Any]:
    manifest_path = seed_dir / "dataset" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "seed": seed,
        "dataset_manifest": manifest_path.as_posix(),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "manifest_assertions": {
            "generator": manifest.get("generator"),
            "caption_graph_equivalent": manifest.get("caption_graph_equivalent"),
        },
    }


def _scientific_interpretation(
    report: dict[str, Any], config: dict[str, Any]
) -> str:
    if config["model"]["backend"] == "reference":
        return (
            "INCONCLUSIVE: ReferenceVLM is a deterministic infrastructure fixture and is not "
            "eligible evidence about structured representation in a real VLM."
        )
    if report["final_decision"] == "GO":
        return "All preregistered information, token, depth, significance, and OOD gates passed."
    if report["final_decision"] == "NO-GO":
        return "At least one falsification gate failed; the structured-interface claim is rejected."
    return "INCONCLUSIVE: model evidence or a preregistered robustness gate is incomplete."


def _record_failed_bundle(
    output: Path,
    config: dict[str, Any],
    seeds: list[int],
    per_seed: list[dict[str, Any]],
    integrity_rows: list[dict[str, Any]],
    error: Exception,
) -> None:
    """Retain failed/incomplete runs instead of silently dropping unfavorable evidence."""

    registration = get_registered_experiment("EXP001")
    result = {
        "schema_version": 1,
        "run_id": f"exp001-failed-{utc_now().replace(':', '').replace('-', '')}",
        "experiment_id": "EXP001",
        "hypothesis": "H001",
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
    run_record = {
        "schema_version": 1,
        "run_id": result["run_id"],
        "experiment_id": "EXP001",
        "status": "failed",
        "decision": "INCONCLUSIVE",
        "completed_at": utc_now(),
        "error": f"{type(error).__name__}: {error}",
        "completed_seeds": [row["seed"] for row in per_seed],
    }
    _write_json(output / "run.json", run_record)
    artifact_names = (
        "config.resolved.yaml",
        "metrics.json",
        "predictions.jsonl",
        "run.json",
        "decision_report.yaml",
        "decision_report.md",
    )
    _write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "experiment_id": "EXP001",
            "run_id": result["run_id"],
            "status": "failed",
            "artifacts": {
                name: {
                    "bytes": (output / name).stat().st_size,
                    "sha256": _sha256(output / name),
                }
                for name in artifact_names
                if (output / name).is_file()
            },
        },
    )


def _write_figures(output: Path, descriptive: dict[str, Any]) -> None:
    condition_names = list(PRIMARY_CONDITIONS)
    condition_values = [
        float(descriptive["condition_summaries"]["natural"][name]["mean"])
        for name in condition_names
    ]
    _bar_chart(
        output / "figure1_performance_comparison.png",
        "EXP001 Performance Comparison (Natural Length)",
        condition_names,
        condition_values,
        "Accuracy",
    )
    depths = ["1", "2", "3", "4"]
    caption = [
        float(descriptive["reasoning_depth_summaries"]["token_matched"][depth]["caption"]["mean"])
        for depth in depths
    ]
    graph = [
        float(
            descriptive["reasoning_depth_summaries"]["token_matched"][depth]["scene_graph"][
                "mean"
            ]
        )
        for depth in depths
    ]
    _line_chart(
        output / "figure2_reasoning_depth.png",
        "Performance vs Reasoning Depth (Token Matched)",
        depths,
        {"Caption": caption, "Graph": graph},
    )
    token_values = [
        float(descriptive["token_efficiency_summaries"][name]["mean"])
        for name in condition_names
    ]
    _bar_chart(
        output / "figure3_token_efficiency.png",
        "Token Efficiency (Natural Length)",
        condition_names,
        token_values,
        "Accuracy / 1,000 input tokens",
        normalized=False,
    )


def _bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    y_label: str,
    *,
    normalized: bool = True,
) -> None:
    image = Image.new("RGB", (960, 600), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 90, 70, 920, 510
    ceiling = 1.0 if normalized else max(values + [1.0]) * 1.1
    _axes(draw, left, top, right, bottom, title, y_label, ceiling)
    slot = (right - left) / len(labels)
    colors = ("#64748b", "#94a3b8", "#f59e0b", "#2563eb")
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        x0 = left + index * slot + slot * 0.18
        x1 = left + (index + 1) * slot - slot * 0.18
        y = bottom - (value / ceiling) * (bottom - top)
        draw.rectangle((x0, y, x1, bottom), fill=colors[index % len(colors)])
        draw.text((x0, bottom + 12), label.replace("_", " "), fill="black")
        draw.text((x0, y - 18), f"{value:.3f}", fill="black")
    image.save(path)


def _line_chart(
    path: Path,
    title: str,
    labels: list[str],
    series: dict[str, list[float]],
) -> None:
    image = Image.new("RGB", (960, 600), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 90, 70, 920, 510
    _axes(draw, left, top, right, bottom, title, "Accuracy", 1.0)
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


def _axes(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    right: int,
    bottom: int,
    title: str,
    y_label: str,
    ceiling: float,
) -> None:
    draw.text((left, 25), title, fill="black")
    draw.text((left, top - 18), y_label, fill="black")
    draw.line((left, top, left, bottom, right, bottom), fill="black", width=2)
    for index in range(6):
        value = ceiling * index / 5
        y = bottom - index * (bottom - top) / 5
        draw.line((left, y, right, y), fill="#e2e8f0", width=1)
        draw.text((45, y - 7), f"{value:.2f}", fill="#475569")


def _bundle_manifest(output: Path, run_record: dict[str, Any]) -> dict[str, Any]:
    names = (
        "config.resolved.yaml",
        "metrics.json",
        "predictions.jsonl",
        "run.json",
        "decision_report.yaml",
        "decision_report.md",
        "figure1_performance_comparison.png",
        "figure2_reasoning_depth.png",
        "figure3_token_efficiency.png",
    )
    return {
        "schema_version": 1,
        "experiment_id": "EXP001",
        "run_id": run_record["run_id"],
        "status": "complete",
        "artifacts": {
            name: {"bytes": (output / name).stat().st_size, "sha256": _sha256(output / name)}
            for name in names
        },
    }


def _ensure_new_bundle(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"EXP001 output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)


def _lookup(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"EXP001 metric path is missing: {path}")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/graph_vs_text.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    result = run(args.config, output_dir=args.output)
    print(f"EXP001 decision: {result['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run", "run_seed_config"]
