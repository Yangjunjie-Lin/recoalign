"""Model × experiment evaluation orchestration without benchmark mutation."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import yaml

from experiments.runtime import checkpoint_provenance, load_config, runtime_provenance
from experiments.unified.runner import run_config
from recoalign.models.vlm.failure_analysis import write_failure_analysis
from recoalign.models.vlm.registry import dry_run_model, get_model_registry
from recoalign.reproducibility import get_git_metadata, utc_now
from recoalign.research_registry import get_registered_experiment, validate_research_registries
from recoalign.schema_validation import repository_root


def run_vlm_evaluation(
    *,
    model_name: str,
    experiment_id: str,
    split: str | None = None,
    seeds: list[int] | None = None,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    cache_enabled: bool | None = None,
    batch_size: int | None = None,
) -> Path:
    """Resolve one registered matrix cell and run or validate it."""

    project = repository_root()
    validate_research_registries(project)
    registration = get_registered_experiment(experiment_id, project)
    source = project / registration["config"]
    experiment_config = load_config(source)
    registry = get_model_registry()
    definition = registry.definition(model_name)
    resolved = registry.apply_to_experiment(experiment_config, definition.name)
    registered_split = str(registration["dataset"]["split"])
    if split is not None and split not in {registered_split, "registered"}:
        raise ValueError(
            f"split override {split!r} would change the registered benchmark split "
            f"{registered_split!r}"
        )
    if seeds is not None:
        _validate_seeds(seeds)
        resolved["experiment"]["seeds"] = list(seeds)
        resolved["experiment"]["seed"] = int(seeds[0])
        resolved["model"]["seed"] = int(seeds[0])
    if cache_enabled is not None:
        resolved["model"]["cache_enabled"] = bool(cache_enabled)
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        resolved["model"]["batch_size"] = batch_size

    output = Path(
        output_dir or project / "outputs" / experiment_id / definition.name
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"VLM evaluation output is not empty: {output}")
    resolved["experiment"]["output_dir"] = output.as_posix()
    resolved.setdefault("vlm_evaluation", {}).update(
        {
            "requested_split": split or "registered",
            "registered_split": registered_split,
            "dry_run": dry_run,
            "matrix_cell": f"{definition.name}×{experiment_id}",
        }
    )

    model_report = dry_run_model(
        resolved["model"], seed=int(resolved["experiment"]["seed"])
    )
    if dry_run:
        _write_dry_run(output, resolved, registration, definition.payload, model_report)
        return output

    selected_seeds = list(resolved["experiment"]["seeds"])
    if seeds is not None and len(selected_seeds) == 1:
        _execute_single_seed(resolved, output, selected_seeds[0])
        result_scope = "single_seed_non_claim"
    else:
        _execute_multiseed(resolved, output, experiment_id)
        result_scope = "registered_multiseed"
    _enrich_run(output, resolved, registration, definition.payload, model_report, result_scope)
    failure_root = (
        project
        / "reports"
        / "failure_analysis"
        / experiment_id
        / definition.name
        / output.name
    )
    failure_summary = write_failure_analysis(output / "predictions.jsonl", failure_root)
    _write_matrix_manifest(output, resolved, model_report, failure_root, failure_summary)
    return output


def compatibility_matrix() -> dict[str, Any]:
    registry = get_model_registry()
    experiments = ("EXP001", "EXP002", "EXP003")
    models: dict[str, Any] = {}
    for name in registry.names():
        definition = registry.definition(name)
        model_config = definition.experiment_model_config()
        dry_run = dry_run_model(model_config)
        models[name] = {
            "adapter": definition.payload["adapter"],
            "scientific_evidence": definition.scientific_evidence,
            "checkpoint_revision": definition.payload["model"].get("revision"),
            "runtime": dry_run,
            "experiments": {
                experiment: {
                    "compatible": experiment in definition.compatibility,
                    "status": (
                        "runtime_ready"
                        if experiment in definition.compatibility
                        and dry_run.get("runtime_ready", dry_run.get("adapter_ready", False))
                        else "adapter_ready"
                        if experiment in definition.compatibility
                        else "unsupported"
                    ),
                }
                for experiment in experiments
            },
        }
    return {
        "schema_version": 1,
        "models": models,
        "experiments": list(experiments),
        "prompt_protocol": "configs/prompts/reasoning_default.yaml",
        "benchmark_modified": False,
    }


def write_compatibility_report(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    matrix = compatibility_matrix()
    _write_json(output / "experiment_compatibility.json", matrix)
    lines = [
        "# Experiment Compatibility Report",
        "",
        "| Model | EXP001 | EXP002 | EXP003 | Claim-eligible after inference |",
        "|---|---|---|---|:---:|",
    ]
    for name, model in matrix["models"].items():
        cells = [model["experiments"][key]["status"] for key in matrix["experiments"]]
        lines.append(
            f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} | "
            f"{'yes' if model['scientific_evidence'] else 'no'} |"
        )
    (output / "experiment_compatibility.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    return matrix


def _execute_single_seed(config: dict[str, Any], output: Path, seed: int) -> None:
    with tempfile.TemporaryDirectory(prefix="recoalign-vlm-") as temporary:
        config_path = Path(temporary) / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=True), encoding="utf-8", newline="\n"
        )
        run_config(config_path, output_dir=output, seed=seed)


def _execute_multiseed(config: dict[str, Any], output: Path, experiment_id: str) -> None:
    with tempfile.TemporaryDirectory(prefix="recoalign-vlm-") as temporary:
        config_path = Path(temporary) / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=True), encoding="utf-8", newline="\n"
        )
        if experiment_id == "EXP001":
            from experiments.graph_vs_text.runner import run
        elif experiment_id == "EXP002":
            from experiments.graph_ablation.runner import run
        elif experiment_id == "EXP003":
            from experiments.ood_composition.runner import run
        else:
            raise ValueError(f"unsupported Phase-1 experiment: {experiment_id}")
        run(config_path, output_dir=output)


def _write_dry_run(
    output: Path,
    config: dict[str, Any],
    registration: dict[str, Any],
    model_definition: dict[str, Any],
    model_report: dict[str, Any],
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    resolved_path = output / "config.resolved.yaml"
    resolved_path.write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8", newline="\n"
    )
    (output / "predictions.jsonl").write_text("", encoding="utf-8")
    metrics = {
        "schema_version": 1,
        "experiment_id": registration["experiment_id"],
        "model": model_definition["name"],
        "status": "dry-run",
        "scientific_decision": "INCONCLUSIVE",
        "reason": "Weights were not loaded and no predictions were generated.",
        "model_validation": model_report,
    }
    _write_json(output / "metrics.json", metrics)
    run = _provenance(config, registration, model_definition, model_report)
    run.update({"status": "dry-run", "prediction_count": 0})
    _write_json(output / "run.json", run)
    _write_json(output / "manifest.json", _artifact_manifest(output, run))


def _enrich_run(
    output: Path,
    config: dict[str, Any],
    registration: dict[str, Any],
    model_definition: dict[str, Any],
    model_report: dict[str, Any],
    result_scope: str,
) -> None:
    path = output / "run.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    existing["vlm_provenance"] = _provenance(
        config, registration, model_definition, model_report
    )
    existing["experiment_id"] = registration["experiment_id"]
    existing["model"] = model_definition["name"]
    existing["result_scope"] = result_scope
    _write_json(path, existing)


def _provenance(
    config: dict[str, Any],
    registration: dict[str, Any],
    model_definition: dict[str, Any],
    model_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "captured_at": utc_now(),
        "experiment_id": registration["experiment_id"],
        "model": model_definition["name"],
        "model_id": model_definition["model"].get("model_id"),
        "revision": model_definition["model"].get("revision"),
        "checkpoint": checkpoint_provenance(config),
        "runtime": runtime_provenance(),
        "generation": dict(config["model"].get("generation", {})),
        "prompt_protocol": config["model"].get("prompt_protocol"),
        "prompt_protocol_sha256": checkpoint_provenance(config).get(
            "prompt_protocol_sha256"
        ),
        "dataset_manifest": registration["dataset"]["manifest"],
        "dataset_manifest_sha256": _sha256(
            repository_root() / registration["dataset"]["manifest"]
        ),
        "git": get_git_metadata(repository_root()),
        "dry_run_validation": model_report,
        "training_enabled": False,
        "benchmark_modified": False,
    }


def _write_matrix_manifest(
    output: Path,
    config: dict[str, Any],
    model_report: dict[str, Any],
    failure_root: Path,
    failure_summary: dict[str, Any],
) -> None:
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    original_manifest_path = output / "manifest.json"
    original_manifest = (
        json.loads(original_manifest_path.read_text(encoding="utf-8"))
        if original_manifest_path.is_file()
        else None
    )
    manifest = _artifact_manifest(output, run)
    manifest.update(
        {
            "model_validation": model_report,
            "cache_enabled": bool(config["model"].get("cache_enabled", False)),
            "batch_size": int(config["model"].get("batch_size", 1)),
            "failure_analysis": {
                "path": failure_root.as_posix(),
                "summary": failure_summary,
            },
            "phase1_manifest": original_manifest,
        }
    )
    _write_json(output / "manifest.json", manifest)


def _artifact_manifest(output: Path, run: dict[str, Any]) -> dict[str, Any]:
    files = {}
    for name in ("config.resolved.yaml", "metrics.json", "predictions.jsonl", "run.json"):
        path = output / name
        if path.is_file():
            files[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return {
        "schema_version": 1,
        "status": run.get("status"),
        "experiment_id": run.get("experiment_id"),
        "model": run.get("model", run.get("model_backend")),
        "artifacts": files,
    }


def _validate_seeds(seeds: list[int]) -> None:
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ValueError("seeds must be unique non-negative integers")


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


__all__ = [
    "compatibility_matrix",
    "run_vlm_evaluation",
    "write_compatibility_report",
]
