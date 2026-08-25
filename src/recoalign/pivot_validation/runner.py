"""Resumable, preregistration-locked execution for PIVOT_EXP_A."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from experiments.runtime import runtime_provenance
from recoalign.models.vlm.registry import ModelRegistry, dry_run_model

from .analysis import analyze_predictions, classify_mechanism
from .design import (
    A1_TASKS,
    A2_CONDITIONS,
    A3_CONDITIONS,
    PivotTrial,
    build_seed_trials,
    load_source_records,
    select_balanced_records,
    validate_seed_trials,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "research/pivot_validation/config.yaml"
FREEZE_PATH = ROOT / "research/pivot_validation/freeze_manifest.yaml"
LOCKED_FILES = (
    "research/pivot_validation/config.yaml",
    "research/pivot_validation/hypothesis_registry.yaml",
    "research/pivot_validation/experiment_registry.yaml",
    "research/pivot_validation/protocols/A1_primitive_semantics.md",
    "research/pivot_validation/protocols/A2_evidence_factorization.md",
    "research/pivot_validation/protocols/A3_format_invariance.md",
    "src/recoalign/pivot_validation/design.py",
    "src/recoalign/pivot_validation/analysis.py",
    "src/recoalign/pivot_validation/runner.py",
)


def load_pivot_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("pivot config must be a mapping")
    _validate_config(payload)
    return payload


def preflight_pivot_validation(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    write_freeze: bool = True,
) -> dict[str, Any]:
    config = load_pivot_config(config_path)
    registry = ModelRegistry(ROOT)
    definition = registry.definition(str(config["model"]["registry_name"]))
    model_config = definition.experiment_model_config()
    model = registry.get_or_create(model_config, seed=int(config["study"]["seeds"][0]))
    seed_reports = []
    dataset_hashes = {}
    for seed in config["study"]["seeds"]:
        source = _source_path(config, int(seed))
        records = select_balanced_records(
            load_source_records(source), int(config["study"]["samples_per_seed"])
        )
        trials = build_seed_trials(
            model,
            records,
            seed=int(seed),
            token_match_tolerance=int(config["A3"]["token_match_tolerance"]),
        )
        report = validate_seed_trials(
            trials, samples_per_seed=int(config["study"]["samples_per_seed"])
        )
        report["seed"] = int(seed)
        report["question_types"] = _question_type_counts(records)
        seed_reports.append(report)
        dataset_hashes[_relative(source)] = _sha256(source)
    dry_run = dry_run_model(model_config, seed=int(config["study"]["seeds"][0]))
    checkpoint = dry_run.get("provenance", {}).get("checkpoint_manifest")
    assertions = {
        "all_seed_designs_pass": all(report["passed"] for report in seed_reports),
        "pilot_is_first_three_final_seeds": config["study"]["pilot_seeds"]
        == config["study"]["seeds"][:3],
        "five_final_seeds": len(config["study"]["seeds"]) == 5,
        "minimum_three_pilot_seeds": len(config["study"]["pilot_seeds"]) >= 3,
        "training_disabled": config["model"]["training_enabled"] is False,
        "model_identifier_matches": config["model"]["identifier"]
        == definition.payload["model"]["model_id"],
        "model_revision_matches": config["model"]["revision"]
        == definition.payload["model"]["revision"],
        "checkpoint_manifest_matches": checkpoint is not None,
        "runtime_ready_without_loading_weights": dry_run.get("runtime_ready") is True
        and dry_run.get("weights_loaded") is False,
    }
    freeze = {
        "schema_version": 1,
        "freeze_name": "pivot_exp_a_v1",
        "status": "preregistered",
        "created_at": "2026-08-21",
        "protocol_changes_allowed_after_inference": False,
        "git": _git_identity(),
        "model": {
            "identifier": config["model"]["identifier"],
            "revision": config["model"]["revision"],
            "config": config["model"]["config"],
            "config_sha256": _sha256(ROOT / config["model"]["config"]),
            "checkpoint_manifest": config["model"]["checkpoint_manifest"],
            "checkpoint_manifest_sha256": _sha256(
                ROOT / config["model"]["checkpoint_manifest"]
            ),
            "prompt_protocol": config["model"]["prompt_protocol"],
            "prompt_protocol_sha256": _sha256(ROOT / config["model"]["prompt_protocol"]),
        },
        "locked_files": {name: _sha256(ROOT / name) for name in LOCKED_FILES},
        "source_datasets": dataset_hashes,
        "design": {
            "seeds": config["study"]["seeds"],
            "pilot_seeds": config["study"]["pilot_seeds"],
            "samples_per_seed": config["study"]["samples_per_seed"],
            "predictions_per_seed": int(config["study"]["samples_per_seed"])
            * (len(A1_TASKS) + len(A2_CONDITIONS) + len(A3_CONDITIONS)),
        },
        "assertions": assertions,
        "passed": all(assertions.values()),
    }
    report = {
        "schema_version": 1,
        "status": "PASS" if freeze["passed"] else "FAIL",
        "weights_loaded": False,
        "freeze": freeze,
        "seed_reports": seed_reports,
        "model_dry_run": dry_run,
    }
    output = _output_root(config)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "preflight.json", report)
    if write_freeze:
        _write_yaml(FREEZE_PATH, freeze)
    if not freeze["passed"]:
        raise RuntimeError("PIVOT_EXP_A preflight failed; inference is prohibited")
    return report


def run_pivot_validation(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    stage: str,
) -> dict[str, Any]:
    if stage not in {"pilot", "final"}:
        raise ValueError("pivot stage must be pilot or final")
    config = load_pivot_config(config_path)
    freeze = _verify_freeze(config)
    seeds = [int(seed) for seed in config["study"]["pilot_seeds" if stage == "pilot" else "seeds"]]
    output = _output_root(config)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "predictions.jsonl"
    existing = _read_jsonl(raw_path)
    by_key = {str(row["trial_key"]): row for row in existing}
    if len(by_key) != len(existing):
        raise ValueError("pivot predictions contain duplicate trial keys")
    registry = ModelRegistry(ROOT)
    definition = registry.definition(str(config["model"]["registry_name"]))
    model_config = definition.experiment_model_config()
    model = registry.get_or_create(model_config, seed=seeds[0])
    expected_per_seed = int(freeze["design"]["predictions_per_seed"])
    started = time.perf_counter()
    new_predictions = 0
    for seed in seeds:
        records = select_balanced_records(
            load_source_records(_source_path(config, seed)),
            int(config["study"]["samples_per_seed"]),
        )
        trials = build_seed_trials(
            model,
            records,
            seed=seed,
            token_match_tolerance=int(config["A3"]["token_match_tolerance"]),
        )
        integrity = validate_seed_trials(
            trials, samples_per_seed=int(config["study"]["samples_per_seed"])
        )
        if not integrity["passed"]:
            raise RuntimeError(f"seed {seed} design integrity failed before inference: {integrity}")
        for trial in trials:
            if trial.key in by_key:
                continue
            prediction = model.reason(
                trial.record,
                trial.condition,
                prepared_input=trial.prepared,
            )
            evaluation = model.evaluate_detailed(prediction, trial.record)
            row = _prediction_row(trial, prediction, evaluation.to_dict(), freeze)
            _append_jsonl(raw_path, row)
            by_key[trial.key] = row
            new_predictions += 1
            if new_predictions % 25 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"PIVOT_EXP_A stage={stage} new={new_predictions} "
                    f"total={len(by_key)} elapsed_s={elapsed:.1f}",
                    flush=True,
                )
        observed = sum(int(row["seed"]) == seed for row in by_key.values())
        if observed != expected_per_seed:
            raise RuntimeError(
                f"seed {seed} prediction count mismatch: {observed} != {expected_per_seed}"
            )
        print(f"PIVOT_EXP_A seed={seed} complete predictions={observed}", flush=True)
    stage_rows = [row for row in by_key.values() if int(row["seed"]) in seeds]
    complete_seeds = [
        seed
        for seed in seeds
        if sum(int(row["seed"]) == seed for row in stage_rows) == expected_per_seed
    ]
    if complete_seeds != seeds:
        raise RuntimeError(f"pivot stage has incomplete seeds: {complete_seeds} vs {seeds}")
    metrics = analyze_predictions(stage_rows, config)
    integrity = _post_run_integrity(stage_rows, config, freeze, seeds)
    metrics["integrity"] = integrity
    metrics["integrity_passed"] = integrity["passed"]
    metrics["mechanism"] = classify_mechanism(metrics, config)
    metrics["stage"] = stage
    metrics["model"] = config["model"]["identifier"]
    metrics["revision"] = config["model"]["revision"]
    metrics["freeze_name"] = freeze["freeze_name"]
    _write_json(output / f"{stage}_metrics.json", metrics)
    _write_json(output / "metrics.json", metrics)
    run = {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A",
        "stage": stage,
        "status": "complete",
        "seeds": seeds,
        "prediction_count": len(stage_rows),
        "new_predictions": new_predictions,
        "elapsed_seconds": time.perf_counter() - started,
        "freeze_name": freeze["freeze_name"],
        "freeze_sha256": _sha256(FREEZE_PATH),
        "git": _git_identity(),
        "runtime": runtime_provenance(),
        "model_provenance": model.provenance(),
        "decision": metrics["mechanism"]["decision"],
    }
    _write_json(output / f"{stage}_run.json", run)
    _write_json(output / "run.json", run)
    if stage == "final":
        _promote_results(config, metrics, run, stage_rows, freeze)
    return {"metrics": metrics, "run": run}


def _prediction_row(
    trial: PivotTrial,
    prediction: str,
    evaluation: dict[str, Any],
    freeze: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A",
        "experiment": trial.experiment,
        "seed": trial.seed,
        "source_scene_id": trial.source_scene_id,
        "trial_id": trial.trial_id,
        "trial_key": trial.key,
        "condition": trial.condition,
        "task": trial.task,
        "question_type": trial.record.metadata.get("question_type"),
        "hop_depth": trial.record.metadata.get("hop_depth"),
        "answer": trial.record.answer,
        "choices": list(trial.record.choices),
        "prediction": prediction,
        "correct": bool(evaluation["correct"]),
        "evaluation": evaluation,
        "input": {
            **trial.prepared.to_dict(),
            "prompt_sha256": hashlib.sha256(trial.prepared.prompt.encode("utf-8")).hexdigest(),
            "image_sha256": trial.record.metadata.get("image_sha256"),
        },
        "metadata": trial.metadata,
        "freeze_name": freeze["freeze_name"],
    }


def _post_run_integrity(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    freeze: dict[str, Any],
    seeds: list[int],
) -> dict[str, Any]:
    expected = int(freeze["design"]["predictions_per_seed"]) * len(seeds)
    assertions = {
        "prediction_count": len(rows) == expected,
        "unique_trial_keys": len({row["trial_key"] for row in rows}) == len(rows),
        "all_seeds_complete": all(
            sum(int(row["seed"]) == seed for row in rows)
            == int(freeze["design"]["predictions_per_seed"])
            for seed in seeds
        ),
        "all_rows_frozen": all(row["freeze_name"] == freeze["freeze_name"] for row in rows),
        "a2_token_matched": _row_token_match(
            rows, "A2", int(config["A2"]["token_match_tolerance"])
        ),
        "a3_token_matched": _row_token_match(
            rows, "A3", int(config["A3"]["token_match_tolerance"])
        ),
        "a3_fact_identical": _row_fact_match(rows, "A3"),
        "locked_files_unchanged": _locked_files_match(freeze),
        "training_disabled": config["model"]["training_enabled"] is False,
    }
    return {"passed": all(assertions.values()), "assertions": assertions, "expected": expected}


def _promote_results(
    config: dict[str, Any],
    metrics: dict[str, Any],
    run: dict[str, Any],
    rows: list[dict[str, Any]],
    freeze: dict[str, Any],
) -> None:
    destination = ROOT / config["study"]["promoted_results"]
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "metrics.json", metrics)
    decision = {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A",
        "status": "complete",
        "decision": metrics["mechanism"]["decision"],
        "mechanism_classification": metrics["mechanism"]["classification"],
        "hypothesis_status": metrics["mechanism"]["hypothesis_status"],
        "reason": metrics["mechanism"]["reason"],
        "evidence": metrics["mechanism"]["evidence"],
        "updated_research_direction": metrics["mechanism"]["updated_research_direction"],
        "paper_writing_allowed": False,
        "model_development_allowed": False,
        "freeze_name": freeze["freeze_name"],
        "seeds": metrics["completed_seeds"],
    }
    _write_yaml(destination / "pivot_decision.yaml", decision)
    _write_yaml(destination / "provenance.yaml", run)
    raw = destination / "predictions.jsonl"
    with raw.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(rows, key=lambda value: str(value["trial_key"])):
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    compressed = destination / "predictions.jsonl.gz"
    with raw.open("rb") as source, compressed.open("wb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as archive:
            shutil.copyfileobj(source, archive)
    raw.unlink()
    report = _render_validation_report(metrics, decision)
    (destination / "PH001_validation_report.md").write_text(
        report, encoding="utf-8", newline="\n"
    )
    artifacts = {}
    for name in (
        "metrics.json",
        "pivot_decision.yaml",
        "provenance.yaml",
        "predictions.jsonl.gz",
        "PH001_validation_report.md",
    ):
        path = destination / name
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    _write_yaml(
        destination / "artifact_manifest.yaml",
        {
            "schema_version": 1,
            "study_id": "PIVOT_EXP_A",
            "decision": decision["decision"],
            "mechanism_classification": decision["mechanism_classification"],
            "prediction_lines": len(rows),
            "freeze_sha256": _sha256(FREEZE_PATH),
            "artifacts": artifacts,
        },
    )


def _render_validation_report(metrics: dict[str, Any], decision: dict[str, Any]) -> str:
    task_rows = "\n".join(
        f"| {task} | {payload['summary']['mean']:.4f} | "
        f"[{payload['summary']['confidence_interval']['lower']:.4f}, "
        f"{payload['summary']['confidence_interval']['upper']:.4f}] | "
        f"{payload['availability']} |"
        for task, payload in metrics["A1"]["tasks"].items()
    )
    a2_rows = "\n".join(
        f"| {name} | {payload['summary']['mean']:.4f} | "
        f"[{payload['summary']['confidence_interval']['lower']:.4f}, "
        f"{payload['summary']['confidence_interval']['upper']:.4f}] | "
        f"{metrics['A2']['stable_effects'][name]} |"
        for name, payload in metrics["A2"]["effects"].items()
    )
    a3_rows = "\n".join(
        f"| {name} | {payload['summary']['mean']:.4f} | "
        f"[{payload['summary']['confidence_interval']['lower']:.4f}, "
        f"{payload['summary']['confidence_interval']['upper']:.4f}] | "
        f"{payload['format_sensitive']} |"
        for name, payload in metrics["A3"]["pairwise"].items()
    )
    evidence = "\n".join(f"- {value}" for value in decision["evidence"]) or "- None"
    return "\n".join(
        [
            "# PH001 Validation Report",
            "",
            "## Decision",
            "",
            f"**{decision['decision']} — {decision['mechanism_classification']}**",
            "",
            decision["reason"],
            "",
            "## A1 Primitive semantic availability",
            "",
            "| Task | Accuracy | 95% CI | Classification |",
            "| --- | ---: | --- | --- |",
            task_rows,
            "",
            "## A2 Evidence factorization",
            "",
            "| Effect | Mean | 95% CI | Stable |",
            "| --- | ---: | --- | ---: |",
            a2_rows,
            "",
            "## A3 Format invariance",
            "",
            f"Classification: **{metrics['A3']['format_classification']}**",
            "",
            "| Pair | Mean difference | 95% CI | Sensitive |",
            "| --- | ---: | --- | ---: |",
            a3_rows,
            "",
            "## Mechanism evidence",
            "",
            evidence,
            "",
            "## Updated direction",
            "",
            decision["updated_research_direction"],
            "",
            "Paper writing and model development remain blocked.",
        ]
    )


def _verify_freeze(config: dict[str, Any]) -> dict[str, Any]:
    if not FREEZE_PATH.is_file():
        raise FileNotFoundError("pivot freeze manifest is missing; run preflight first")
    freeze = yaml.safe_load(FREEZE_PATH.read_text(encoding="utf-8"))
    if not isinstance(freeze, dict) or freeze.get("passed") is not True:
        raise ValueError("pivot freeze manifest is not a passing preregistration")
    if not _locked_files_match(freeze):
        raise ValueError("pivot protocol/code drift detected after preregistration")
    for path, digest in freeze["source_datasets"].items():
        if _sha256(ROOT / path) != digest:
            raise ValueError(f"pivot source dataset drift detected: {path}")
    if list(freeze["design"]["seeds"]) != list(config["study"]["seeds"]):
        raise ValueError("pivot seed set differs from preregistration")
    return freeze


def _locked_files_match(freeze: dict[str, Any]) -> bool:
    return all(_sha256(ROOT / path) == digest for path, digest in freeze["locked_files"].items())


def _row_token_match(rows: list[dict[str, Any]], experiment: str, tolerance: int) -> bool:
    grouped: dict[tuple[int, str], list[int]] = {}
    for row in rows:
        if row["experiment"] == experiment:
            grouped.setdefault((int(row["seed"]), str(row["source_scene_id"])), []).append(
                int(row["input"]["input_tokens"])
            )
    return bool(grouped) and all(
        max(values) - min(values) <= tolerance for values in grouped.values()
    )


def _row_fact_match(rows: list[dict[str, Any]], experiment: str) -> bool:
    grouped: dict[tuple[int, str], set[str]] = {}
    for row in rows:
        if row["experiment"] == experiment:
            grouped.setdefault((int(row["seed"]), str(row["source_scene_id"])), set()).add(
                str(row["input"]["semantic_facts_sha256"])
            )
    return bool(grouped) and all(len(values) == 1 for values in grouped.values())


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "study",
        "model",
        "A1",
        "A2",
        "A3",
        "statistics",
        "final_decision",
        "prohibitions",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"pivot config missing keys: {missing}")
    if int(config["schema_version"]) != 1:
        raise ValueError("pivot config schema_version must be 1")
    seeds = config["study"]["seeds"]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("pivot final protocol requires five unique seeds")
    if config["study"]["pilot_seeds"] != seeds[:3]:
        raise ValueError("pivot pilot must use the first three final seeds")
    if int(config["study"]["samples_per_seed"]) % 4:
        raise ValueError("pivot samples_per_seed must preserve four-way question balance")
    if config["model"]["training_enabled"] is not False:
        raise ValueError("pivot validation prohibits training")
    if not all(bool(value) for value in config["prohibitions"].values()):
        raise ValueError("all pivot prohibitions must remain enabled")


def _source_path(config: dict[str, Any], seed: int) -> Path:
    return ROOT / str(config["study"]["source_dataset_template"]).format(seed=seed)


def _output_root(config: dict[str, Any]) -> Path:
    return ROOT / config["study"]["output_root"]


def _question_type_counts(records: list[Any]) -> dict[str, int]:
    return {
        name: sum(str(record.metadata.get("question_type")) == name for record in records)
        for name in ("multi_hop", "relation_reasoning", "object_reasoning", "attribute_reasoning")
    }


def _git_identity() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, check=False, text=True
        )
        return result.stdout.strip()

    status = run("status", "--porcelain=v1", "--untracked-files=all")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=ROOT, capture_output=True, check=False
    ).stdout
    untracked = run("ls-files", "--others", "--exclude-standard")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_files": untracked.splitlines() if untracked else [],
    }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )


__all__ = [
    "DEFAULT_CONFIG",
    "FREEZE_PATH",
    "load_pivot_config",
    "preflight_pivot_validation",
    "run_pivot_validation",
]
