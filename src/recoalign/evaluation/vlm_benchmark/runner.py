"""One BaseVLM × method × benchmark × seed evaluation cell."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from recoalign.models.vlm import BaseVLM
from recoalign.models.vlm.registry import dry_run_model, get_model_registry
from recoalign.reproducibility import (
    atomic_write_json,
    collect_environment,
    get_git_metadata,
    seed_everything,
    utc_now,
)

from .adapters import inspect_benchmark_availability, load_benchmark_dataset
from .methods import generate_prediction, method_readiness, validate_method
from .metrics import compute_benchmark_metrics


def load_benchmark_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("benchmark evaluation config root must be a mapping")
    validate_benchmark_config(payload)
    return payload


def validate_benchmark_config(config: dict[str, Any]) -> None:
    required = ("model", "method", "benchmark", "seed", "generation", "prompt", "output_root")
    missing = [field for field in required if field not in config]
    if missing:
        raise ValueError(f"benchmark evaluation config is missing: {', '.join(missing)}")
    if not isinstance(config["model"], dict) or not str(config["model"].get("name", "")):
        raise ValueError("benchmark model requires a registered name")
    if not isinstance(config["method"], dict):
        raise ValueError("benchmark method must be a mapping")
    validate_method(str(config["method"].get("name", "")))
    benchmark = config["benchmark"]
    if not isinstance(benchmark, dict):
        raise ValueError("benchmark must be a mapping")
    for field in (
        "name",
        "category",
        "format",
        "manifest",
        "annotation_file",
        "image_root",
        "split",
    ):
        if not str(benchmark.get(field, "")).strip():
            raise ValueError(f"benchmark.{field} is required")
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("benchmark seed must be a non-negative integer")
    generation = config["generation"]
    if not isinstance(generation, dict):
        raise ValueError("generation must be a mapping")
    if bool(generation.get("do_sample", False)) or float(generation.get("temperature", 0)) != 0:
        raise ValueError("fair evaluation requires deterministic do_sample=false and temperature=0")
    if int(generation.get("max_new_tokens", 0)) <= 0:
        raise ValueError("generation.max_new_tokens must be positive")
    statistics = config.get("statistics", {})
    if not isinstance(statistics, dict) or int(statistics.get("minimum_seeds", 3)) < 3:
        raise ValueError("statistics.minimum_seeds must be at least 3")
    locked = config.get("protocol_lock", {})
    if not isinstance(locked, dict) or locked.get("allow_split_override") is not False:
        raise ValueError("protocol_lock.allow_split_override must be false")


def run_benchmark_cell(
    config: str | Path | dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    model: BaseVLM | None = None,
    dry_run: bool = False,
    project_root: str | Path | None = None,
    capture_environment_metadata: bool = True,
) -> Path:
    resolved = load_benchmark_config(config) if isinstance(config, (str, Path)) else dict(config)
    validate_benchmark_config(resolved)
    root = Path(project_root or Path.cwd()).resolve()
    model_name = str(resolved["model"]["name"])
    method_name = str(resolved["method"]["name"])
    benchmark_name = str(resolved["benchmark"]["name"])
    seed = int(resolved["seed"])
    destination = Path(
        output_dir
        or Path(str(resolved["output_root"]))
        / model_name
        / benchmark_name
        / method_name
        / f"seed_{seed}"
    )
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"benchmark output is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    seed_everything(seed, deterministic=True)
    registry = get_model_registry()
    definition = registry.definition(model_name)
    model_config = definition.experiment_model_config()
    model_report = (
        {
            "adapter": model.model_id,
            "adapter_ready": True,
            "runtime_ready": True,
            "weights_loaded": model.loaded,
            "injected_for_test": True,
            "provenance": model.provenance(),
        }
        if model is not None
        else dry_run_model(model_config, seed=seed)
    )
    availability = inspect_benchmark_availability(resolved, project_root=root)
    if availability.get("format") == "registered_experiment" and not dry_run:
        raise RuntimeError(
            "mechanism cells must be executed through run-vlm-eval/run-experiment; "
            "the comprehensive runner only audits their registered coverage"
        )
    sample = None
    dataset = None
    if availability["available"] and availability.get("format") != "registered_experiment":
        dataset = load_benchmark_dataset(resolved, project_root=root)
        sample = dataset.samples[0]
    readiness = method_readiness(method_name, resolved, sample=sample, model=model)
    blockers = []
    if not availability["available"]:
        blockers.append("benchmark data or manifest is unavailable")
    if model is None and not bool(model_report.get("runtime_ready", False)):
        blockers.append("registered model runtime/checkpoint is not ready")
    if not readiness.ready:
        blockers.append(readiness.reason or "method is not ready")
    _write_common_artifacts(
        destination,
        resolved,
        definition.payload,
        model_report,
        availability,
        readiness.__dict__,
        root,
        capture_environment_metadata,
    )
    if dry_run:
        _write_dry_run(destination, resolved, blockers)
        return destination
    if blockers:
        _write_blocked(destination, resolved, blockers)
        raise RuntimeError("benchmark cell is blocked: " + "; ".join(blockers))
    assert dataset is not None
    active_model = model or registry.get_or_create(model_config, seed=seed)
    active_model.ensure_loaded()
    predictions = []
    for evaluation_sample in dataset.samples:
        prediction = generate_prediction(active_model, evaluation_sample, method_name, resolved)
        scored = active_model.answer_evaluator.evaluate(
            prediction,
            evaluation_sample.answer,
            choices=evaluation_sample.choices,
            allow_reasoning_trace=True,
        )
        predictions.append(
            {
                "sample_id": evaluation_sample.sample_id,
                "group_id": evaluation_sample.group_id,
                "benchmark": dataset.name,
                "model": model_name,
                "method": method_name,
                "seed": seed,
                "prediction": prediction,
                "ground_truth": evaluation_sample.answer,
                "correct": scored.correct,
                "evaluation_method": scored.method,
                "category": evaluation_sample.category,
                "dimension": evaluation_sample.dimension,
                "tags": list(evaluation_sample.tags),
                "distribution": evaluation_sample.metadata.get("distribution"),
                "failure_type": (
                    None if scored.correct else _failure_type(evaluation_sample.dimension)
                ),
                "oracle_graph_at_inference": readiness.oracle_graph_at_inference,
            }
        )
    _write_jsonl(destination / "predictions.jsonl", predictions)
    metrics = compute_benchmark_metrics(predictions, aggregation=dataset.aggregation)
    metrics.update(
        {
            "schema_version": 1,
            "status": "complete",
            "model": model_name,
            "method": method_name,
            "benchmark": benchmark_name,
            "seed": seed,
        }
    )
    atomic_write_json(destination / "metrics.json", metrics)
    report = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "matrix_cell": f"{model_name}×{method_name}×{benchmark_name}×{seed}",
        "prediction_count": len(predictions),
        "scientific_evidence_eligible": bool(definition.scientific_evidence)
        and not bool(model_report.get("injected_for_test", False)),
        "oracle_graph_at_inference": readiness.oracle_graph_at_inference,
        "benchmark_modified": False,
        "prompt_modified_per_model": False,
        "metrics_file": "metrics.json",
        "predictions_file": "predictions.jsonl",
    }
    atomic_write_json(destination / "report.json", report)
    return destination


def _write_common_artifacts(
    destination: Path,
    config: dict[str, Any],
    model_definition: dict[str, Any],
    model_report: dict[str, Any],
    availability: dict[str, Any],
    readiness: dict[str, Any],
    project_root: Path,
    capture_environment_metadata: bool,
) -> None:
    (destination / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    environment = (
        collect_environment(project_root)
        if capture_environment_metadata
        else {
            "schema_version": 1,
            "capture_skipped": True,
            "reason": "fixture or dry-run validation",
            "git": get_git_metadata(project_root),
        }
    )
    atomic_write_json(destination / "environment.json", environment)
    checkpoint_manifest = {
        "schema_version": 1,
        "model": model_definition.get("name"),
        "model_id": model_definition.get("model", {}).get("model_id"),
        "revision": model_definition.get("model", {}).get("revision"),
        "backbone_manifest": model_definition.get("checkpoint_manifest"),
        "interface_checkpoint": readiness.get("checkpoint"),
        "interface_checkpoint_sha256": _optional_sha256(readiness.get("checkpoint")),
        "model_validation": model_report,
    }
    (destination / "checkpoint_manifest.yaml").write_text(
        yaml.safe_dump(checkpoint_manifest, sort_keys=True), encoding="utf-8"
    )
    atomic_write_json(destination / "availability.json", availability)


def _write_dry_run(destination: Path, config: dict[str, Any], blockers: list[str]) -> None:
    (destination / "predictions.jsonl").write_text("", encoding="utf-8")
    status = "ready" if not blockers else "blocked"
    metrics = {
        "schema_version": 1,
        "status": f"dry_run_{status}",
        "scientific_decision": "INCONCLUSIVE",
        "reason": "No inference was executed.",
        "blockers": blockers,
    }
    atomic_write_json(destination / "metrics.json", metrics)
    atomic_write_json(
        destination / "report.json",
        {
            "schema_version": 1,
            "status": f"dry_run_{status}",
            "matrix_cell": _cell(config),
            "blockers": blockers,
            "prediction_count": 0,
            "scientific_evidence_eligible": False,
            "benchmark_modified": False,
        },
    )


def _write_blocked(destination: Path, config: dict[str, Any], blockers: list[str]) -> None:
    (destination / "predictions.jsonl").write_text("", encoding="utf-8")
    atomic_write_json(
        destination / "metrics.json",
        {"schema_version": 1, "status": "blocked", "blockers": blockers},
    )
    atomic_write_json(
        destination / "report.json",
        {
            "schema_version": 1,
            "status": "blocked",
            "matrix_cell": _cell(config),
            "blockers": blockers,
            "prediction_count": 0,
            "scientific_evidence_eligible": False,
        },
    )


def _cell(config: dict[str, Any]) -> str:
    return "×".join(
        (
            str(config["model"]["name"]),
            str(config["method"]["name"]),
            str(config["benchmark"]["name"]),
            str(config["seed"]),
        )
    )


def _failure_type(dimension: str) -> str:
    return {
        "relation": "relation_confusion",
        "attribute": "attribute_swap",
        "object": "object_binding",
    }.get(dimension, "composition_failure")


def _optional_sha256(value: Any) -> str | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            encoded = json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write(encoded + "\n")


__all__ = [
    "load_benchmark_config",
    "run_benchmark_cell",
    "validate_benchmark_config",
]
