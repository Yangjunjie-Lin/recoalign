"""Registry-driven execution, provenance capture, aggregation, and scientific decisions."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from evaluation.statistics import multiple_seed_summary
from experiments.unified.runner import run_config
from recoalign.reproducibility import (
    atomic_write_json,
    collect_environment,
    get_git_metadata,
    utc_now,
)
from recoalign.research_registry import (
    EXPERIMENT_REGISTRY,
    HYPOTHESIS_REGISTRY,
    get_registered_experiment,
    validate_research_registries,
)
from recoalign.schema_validation import repository_root, validate_payload
from research.decision_engine import evaluate_decision, render_decision_report


def run_registered_experiment(
    experiment_id: str,
    *,
    config_path: str | Path | None = None,
    output_root: str | Path = "runs",
    run_id: str | None = None,
    dry_run: bool = False,
    seeds: list[int] | None = None,
    command: str | None = None,
) -> Path:
    """Materialize and optionally execute one experiment registered to a hypothesis."""

    project = repository_root()
    validate_research_registries(project)
    registration = get_registered_experiment(experiment_id, project)
    source_config = _resolve(project, config_path or registration["config"])
    registered_config = _resolve(project, registration["config"])
    if source_config.resolve() != registered_config.resolve():
        raise ValueError(
            "configuration is not preregistered for this experiment; update the registry first"
        )
    config = _load_structured_config(source_config)
    _verify_config_binding(config, registration)
    selected_seeds = list(seeds or config["experiment"]["seeds"])
    _validate_seeds(selected_seeds)

    identifier = run_id or _new_run_id()
    root = _resolve(project, output_root)
    run_dir = root / experiment_id / identifier
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    resolved = deepcopy(config)
    resolved["experiment"]["output_dir"] = str(run_dir.as_posix())
    resolved["experiment"]["seeds"] = selected_seeds
    resolved.setdefault("governance", {}).update(
        {
            "dry_run": dry_run,
            "run_id": identifier,
            "experiment_registry": str(EXPERIMENT_REGISTRY.as_posix()),
            "hypothesis_registry": str(HYPOTHESIS_REGISTRY.as_posix()),
        }
    )
    resolved_path = run_dir / "config.resolved.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")

    git = get_git_metadata(project)
    environment = collect_environment(project)
    (run_dir / "command.txt").write_text(
        (command or _default_command(experiment_id, source_config, dry_run)) + "\n",
        encoding="utf-8",
    )
    (run_dir / "environment.txt").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "git_commit.txt").write_text(
        _render_git_commit(git), encoding="utf-8"
    )
    (run_dir / "seed.txt").write_text(
        "\n".join(str(seed) for seed in selected_seeds) + "\n", encoding="utf-8"
    )
    log_path = run_dir / "log.txt"
    _append_log(log_path, "DRY_RUN_INITIALIZED" if dry_run else "RUN_INITIALIZED")

    manifest = _initial_manifest(
        registration,
        identifier,
        source_config,
        resolved_path,
        selected_seeds,
        git,
        project,
        config,
        dry_run,
    )
    atomic_write_json(run_dir / "manifest.json", manifest)

    result = _result_envelope(registration, config, identifier, selected_seeds, dry_run)
    if dry_run:
        validate_payload("experiment_result", result)
        atomic_write_json(run_dir / "metrics.json", result)
        decision_report = evaluate_decision(registration, result)
        validate_payload("decision_report", decision_report)
        _write_decision_reports(run_dir, decision_report)
        _append_log(log_path, "DRY_RUN_COMPLETE")
        return run_dir

    try:
        per_seed_metrics: list[dict[str, Any]] = []
        integrity_rows: list[dict[str, Any]] = []
        for seed in selected_seeds:
            seed_dir = run_dir / "seeds" / str(seed)
            _append_log(log_path, f"SEED_STARTED seed={seed}")
            metrics = run_config(source_config, output_dir=seed_dir, seed=seed)
            per_seed_metrics.append({"seed": seed, "metrics": metrics})
            integrity_rows.append(_seed_integrity(seed_dir, seed, registration))
            _append_log(log_path, f"SEED_COMPLETE seed={seed}")

        aggregate = _aggregate_registered_metrics(registration, per_seed_metrics, config)
        result["metrics"] = {"per_seed": per_seed_metrics, "aggregate": aggregate}
        result["confidence_interval"] = {
            path: summary["confidence_interval"] for path, summary in aggregate.items()
        }
        result["statistical_test"] = {
            path: summary["statistical_test"] for path, summary in aggregate.items()
        }
        result["integrity"] = {
            "registry_validated": True,
            "dataset_version": registration["dataset"]["version"],
            "per_seed": integrity_rows,
        }
        result["status"] = "complete"
        decision_report = evaluate_decision(registration, result)
        validate_payload("decision_report", decision_report)
        result["decision"] = decision_report["final_decision"]
        result["timestamp"] = utc_now()
        validate_payload("experiment_result", result)
        atomic_write_json(run_dir / "metrics.json", result)
        _write_decision_reports(run_dir, decision_report)
        manifest["status"] = "complete"
        manifest["completed_at"] = utc_now()
        manifest["decision"] = result["decision"]
        atomic_write_json(run_dir / "manifest.json", manifest)
        _append_log(log_path, f"RUN_COMPLETE decision={result['decision']}")
        return run_dir
    except Exception as exc:
        result["status"] = "failed"
        result["decision"] = "INCONCLUSIVE"
        result["timestamp"] = utc_now()
        atomic_write_json(run_dir / "metrics.json", result)
        manifest["status"] = "failed"
        manifest["completed_at"] = utc_now()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(run_dir / "manifest.json", manifest)
        _append_log(log_path, f"RUN_FAILED error={type(exc).__name__}: {exc}")
        raise


def decide_existing_result(
    result_path: str | Path, *, output: str | Path | None = None
) -> dict[str, Any]:
    """Reapply the committed rule and record a schema-valid YAML decision report."""

    path = Path(result_path)
    result = json.loads(path.read_text(encoding="utf-8"))
    validate_payload("experiment_result", result)
    experiment = get_registered_experiment(result["experiment_id"])
    report = evaluate_decision(experiment, result)
    validate_payload("decision_report", report)
    destination = Path(output) if output else path.with_name("decision_report.yaml")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    return report


def _aggregate_registered_metrics(
    registration: dict[str, Any],
    per_seed: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    paths = [criterion["metric_path"] for criterion in registration["decision_rule"]["criteria"]]
    aggregate: dict[str, Any] = {}
    evaluation = config["evaluation"]
    for index, path in enumerate(paths):
        values = [float(_lookup(row["metrics"], path)) for row in per_seed]
        aggregate[path] = multiple_seed_summary(
            values,
            confidence=float(evaluation["confidence_level"]),
            bootstrap_samples=int(evaluation["bootstrap_samples"]),
            seed=int(evaluation["statistics_seed"]) + index,
            significance_test=str(evaluation.get("significance_test", "paired_t_test")),
        )
    return aggregate


def _seed_integrity(
    seed_dir: Path, seed: int, registration: dict[str, Any]
) -> dict[str, Any]:
    assertions: dict[str, Any] = {}
    required = registration["decision_rule"].get("required_manifest_assertions", {})
    dataset_manifest = seed_dir / "dataset" / "manifest.json"
    payload: dict[str, Any] = {}
    run_payload: dict[str, Any] = {}
    if dataset_manifest.is_file():
        payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))
        if required:
            assertions = {key: payload.get(key) for key in required}
    run_path = seed_dir / "run.json"
    if run_path.is_file():
        run_payload = json.loads(run_path.read_text(encoding="utf-8"))
        assertions.update(dict(run_payload.get("manifest_assertions", {})))
    return {
        "seed": seed,
        "run_manifest": str((seed_dir / "run.json").as_posix()),
        "dataset_manifest": str(dataset_manifest.as_posix()),
        "dataset_manifest_sha256": (
            _sha256(dataset_manifest) if dataset_manifest.is_file() else None
        ),
        "dataset_generator": payload.get("generator"),
        "manifest_assertions": assertions,
    }


def _result_envelope(
    registration: dict[str, Any],
    config: dict[str, Any],
    run_id: str,
    seeds: list[int],
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "experiment_id": registration["experiment_id"],
        "hypothesis": registration["hypothesis_id"],
        "model": f"frozen-{config['model']['backend']}",
        "dataset": f"{registration['dataset']['name']}@{registration['dataset']['version']}",
        "seed": seeds,
        "metrics": {},
        "confidence_interval": None,
        "statistical_test": None,
        "decision": "INCONCLUSIVE",
        "timestamp": utc_now(),
        "integrity": {"registry_validated": True},
        "status": "dry-run" if dry_run else "complete",
    }


def _initial_manifest(
    registration: dict[str, Any],
    run_id: str,
    source_config: Path,
    resolved_path: Path,
    seeds: list[int],
    git: dict[str, Any],
    project: Path,
    config: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "experiment_id": registration["experiment_id"],
        "hypothesis_id": registration["hypothesis_id"],
        "status": "dry-run" if dry_run else "running",
        "started_at": utc_now(),
        "completed_at": utc_now() if dry_run else None,
        "code_version": git,
        "config": {
            "source": str(source_config.as_posix()),
            "resolved": str(resolved_path.as_posix()),
            "sha256": _sha256(resolved_path),
        },
        "dataset": {
            **registration["dataset"],
            "manifest_sha256": _sha256(project / registration["dataset"]["manifest"]),
        },
        "model": {
            **registration["model"],
            "checkpoint_manifest": config["model"].get("checkpoint_manifest"),
            "checkpoint_manifest_sha256": _optional_manifest_sha256(
                project, config["model"].get("checkpoint_manifest")
            ),
        },
        "seeds": seeds,
        "registries": {
            "hypotheses": {
                "path": str(HYPOTHESIS_REGISTRY.as_posix()),
                "sha256": _sha256(project / HYPOTHESIS_REGISTRY),
            },
            "experiments": {
                "path": str(EXPERIMENT_REGISTRY.as_posix()),
                "sha256": _sha256(project / EXPERIMENT_REGISTRY),
            },
        },
        "protocol": registration["protocol"],
        "files": {
            "config": "config.resolved.yaml",
            "command": "command.txt",
            "environment": "environment.txt",
            "git_commit": "git_commit.txt",
            "seed": "seed.txt",
            "metrics": "metrics.json",
            "log": "log.txt",
            "decision_report": "decision_report.yaml",
            "decision_report_markdown": "decision_report.md",
        },
    }


def _write_decision_reports(run_dir: Path, report: dict[str, Any]) -> None:
    (run_dir / "decision_report.yaml").write_text(
        yaml.safe_dump(report, sort_keys=False), encoding="utf-8"
    )
    (run_dir / "decision_report.md").write_text(
        render_decision_report(report), encoding="utf-8"
    )


def _verify_config_binding(config: dict[str, Any], registration: dict[str, Any]) -> None:
    experiment = config["experiment"]
    if experiment.get("id") != registration["experiment_id"]:
        raise ValueError("configuration experiment.id does not match registration")
    if experiment.get("hypothesis_id") != registration["hypothesis_id"]:
        raise ValueError("configuration hypothesis link does not match registration")
    if config.get("training", {}).get("enabled") is not False:
        raise ValueError("governed experiments must set training.enabled=false")


def _load_structured_config(path: Path) -> dict[str, Any]:
    from experiments.runtime import load_config

    return load_config(path)


def _validate_seeds(seeds: list[int]) -> None:
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ValueError("seeds must be unique non-negative integers")


def _lookup(payload: dict[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"registered metric path is missing from result: {dotted_path}")
        value = value[part]
    return value


def _resolve(project: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project / candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_manifest_sha256(project: Path, value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = _resolve(project, value)
    return _sha256(path) if path.is_file() else None


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _default_command(experiment_id: str, config: Path, dry_run: bool) -> str:
    suffix = " --dry-run" if dry_run else ""
    return f"recoalign run-experiment {experiment_id} --config {config.as_posix()}{suffix}"


def _render_git_commit(git: dict[str, Any]) -> str:
    return (
        f"commit={git.get('commit')}\n"
        f"branch={git.get('branch')}\n"
        f"dirty={git.get('dirty')}\n"
        f"diff_sha256={git.get('diff_sha256')}\n"
        f"untracked_count={git.get('untracked_count')}\n"
    )


def _append_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_now()} {message}\n")
