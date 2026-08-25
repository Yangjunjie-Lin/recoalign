"""Single lifecycle runner for preregistration, preflight, inference, and adjudication."""

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

from .decision import adjudicate_mechanism
from .factorial_builder import (
    CausalTrial,
    build_seed_trials,
    design_digest,
    load_source_records,
    validate_seed_trials,
)
from .integrity import (
    artifact_metadata,
    locked_files_match,
    source_files_match,
    validate_parent_freeze,
    validate_prediction_rows,
)
from .power import build_power_analysis
from .statistics import analyze_causal_predictions

ROOT = Path(__file__).resolve().parents[3]
STUDY_ROOT = ROOT / "research/causal_separation/PIVOT_EXP_A2"
DEFAULT_CONFIG = STUDY_ROOT / "config.yaml"
FREEZE_PATH = STUDY_ROOT / "freeze_manifest.yaml"
FROZEN_PARENT_PREDICTIONS = ROOT / "research/pivot_validation/results/predictions.jsonl.gz"
LOCKED_FILES = (
    "research/causal_separation/PIVOT_EXP_A2/hypothesis_registry.yaml",
    "research/causal_separation/PIVOT_EXP_A2/preregistration.md",
    "research/causal_separation/PIVOT_EXP_A2/power_analysis.yaml",
    "research/causal_separation/PIVOT_EXP_A2/config.yaml",
    "research/causal_separation/PIVOT_EXP_A2/decision_policy.yaml",
    "research/causal_separation/PIVOT_EXP_A2/serializers/README.md",
    "research/causal_separation/PIVOT_EXP_A2/protocols/factorial_protocol.md",
    "research/causal_separation/PIVOT_EXP_A2/protocols/manipulation_check.md",
    "src/recoalign/causal_separation/semantic_scaffold.py",
    "src/recoalign/causal_separation/relation_intervention.py",
    "src/recoalign/causal_separation/serializers.py",
    "src/recoalign/causal_separation/factorial_builder.py",
    "src/recoalign/causal_separation/power.py",
    "src/recoalign/causal_separation/statistics.py",
    "src/recoalign/causal_separation/decision.py",
    "src/recoalign/causal_separation/integrity.py",
    "src/recoalign/causal_separation/runner.py",
)


def load_causal_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("causal-separation config must be a mapping")
    _validate_config(payload)
    return payload


def preregister_causal_separation(
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_causal_config(config_path)
    parent = validate_parent_freeze(ROOT, config["study"]["parent_freeze"])
    power = build_power_analysis(
        FROZEN_PARENT_PREDICTIONS,
        planned_scenes=int(config["study"]["primary_scenes_per_seed"])
        * len(config["study"]["seeds"]),
        superiority_sesoi=float(config["statistics"]["superiority_sesoi"]),
        equivalence_margin=float(config["statistics"]["equivalence_margin"]),
        alpha=float(config["statistics"]["alpha"]),
        target_power=float(config["statistics"]["target_power"]),
    )
    _write_yaml(STUDY_ROOT / "power_analysis.yaml", power)
    report = {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A2",
        "status": "PASS" if parent["passed"] and power["status"] == "PASS" else "FAIL",
        "parent_freeze": parent,
        "power": power,
        "inference_started": False,
        "weights_loaded": False,
    }
    _write_json(STUDY_ROOT / "validation/preregistration_report.json", report)
    if report["status"] != "PASS":
        raise RuntimeError("causal-separation preregistration or power analysis failed")
    return report


def validate_causal_separation(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    preflight_only: bool = True,
    write_freeze: bool = True,
) -> dict[str, Any]:
    del preflight_only
    preregistration = preregister_causal_separation(config_path)
    config = load_causal_config(config_path)
    registry = ModelRegistry(ROOT)
    definition = registry.definition(str(config["model"]["registry_name"]))
    model_config = definition.experiment_model_config()
    model = registry.get_or_create(model_config, seed=int(config["study"]["seeds"][0]))
    seed_reports = []
    dataset_hashes = {}
    design_hashes = {}
    for seed_value in config["study"]["seeds"]:
        seed = int(seed_value)
        source = _source_path(config, seed)
        trials = build_seed_trials(
            model,
            load_source_records(source),
            seed=seed,
            alias_salt=str(config["factorial_design"]["alias_randomization_salt"]),
            corruption_salt=str(
                config["factorial_design"]["corruption_randomization_salt"]
            ),
            token_match_tolerance=int(
                config["factorial_design"]["token_match_tolerance"]
            ),
        )
        report = validate_seed_trials(
            trials, expected_per_seed=int(config["study"]["predictions_per_seed"])
        )
        report["seed"] = seed
        seed_reports.append(report)
        dataset_hashes[_relative(source)] = _sha256(source)
        design_hashes[str(seed)] = design_digest(trials)
    dry_run = dry_run_model(model_config, seed=int(config["study"]["seeds"][0]))
    assertions = {
        "preregistration_passed": preregistration["status"] == "PASS",
        "parent_freeze_passed": preregistration["parent_freeze"]["passed"],
        "power_passed": preregistration["power"]["status"] == "PASS",
        "all_seed_designs_pass": all(report["passed"] for report in seed_reports),
        "five_seeds": len(config["study"]["seeds"]) == 5,
        "predictions_per_seed": int(config["study"]["predictions_per_seed"]) == 1060,
        "training_disabled": config["model"]["training_enabled"] is False,
        "model_identifier_matches": config["model"]["identifier"]
        == definition.payload["model"]["model_id"],
        "model_revision_matches": config["model"]["revision"]
        == definition.payload["model"]["revision"],
        "runtime_ready_without_weights": dry_run.get("runtime_ready") is True
        and dry_run.get("weights_loaded") is False,
    }
    freeze = {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A2",
        "freeze_name": "pivot_exp_a2_v1",
        "status": "preregistered",
        "created_at": "2026-08-21",
        "inference_started": False,
        "protocol_changes_allowed_after_inference": False,
        "parent_freeze": preregistration["parent_freeze"],
        "power_analysis_sha256": _sha256(STUDY_ROOT / "power_analysis.yaml"),
        "model": {
            "identifier": config["model"]["identifier"],
            "revision": config["model"]["revision"],
            "config_sha256": _sha256(ROOT / config["model"]["config"]),
            "checkpoint_manifest_sha256": _sha256(
                ROOT / config["model"]["checkpoint_manifest"]
            ),
            "prompt_protocol_sha256": _sha256(ROOT / config["model"]["prompt_protocol"]),
        },
        "locked_files": {relative: _sha256(ROOT / relative) for relative in LOCKED_FILES},
        "source_datasets": dataset_hashes,
        "design_sha256_by_seed": design_hashes,
        "design": {
            "seeds": config["study"]["seeds"],
            "primary_scenes": int(config["study"]["primary_scenes_per_seed"]) * 5,
            "main_predictions": int(config["study"]["primary_scenes_per_seed"]) * 5 * 8,
            "reference_predictions": int(config["study"]["primary_scenes_per_seed"]) * 5 * 2,
            "manipulation_predictions": int(config["study"]["manipulation_scenes_per_seed"])
            * 5
            * 8,
            "total_predictions": int(config["study"]["predictions_per_seed"]) * 5,
        },
        "assertions": assertions,
        "passed": all(assertions.values()),
    }
    report = {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A2",
        "status": "PASS" if freeze["passed"] else "FAIL",
        "weights_loaded": False,
        "inference_started": False,
        "freeze": freeze,
        "seed_reports": seed_reports,
        "model_dry_run": dry_run,
    }
    _write_json(STUDY_ROOT / "validation/preflight.json", report)
    if write_freeze:
        _write_yaml(FREEZE_PATH, freeze)
    if report["status"] != "PASS":
        raise RuntimeError("PIVOT_EXP_A2 preflight failed; inference is prohibited")
    return report


def run_causal_separation(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    model_name: str = "llava_1_5_7b",
) -> dict[str, Any]:
    config = load_causal_config(config_path)
    if model_name != config["model"]["registry_name"]:
        raise ValueError("causal-separation run must use the preregistered model")
    freeze = _verify_freeze(config)
    output = _output_root(config)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "predictions.jsonl"
    existing = _read_jsonl(raw_path)
    by_key = {str(row["trial_key"]): row for row in existing}
    if len(by_key) != len(existing):
        raise ValueError("causal-separation predictions contain duplicate trial keys")
    registry = ModelRegistry(ROOT)
    definition = registry.definition(model_name)
    model = registry.get_or_create(
        definition.experiment_model_config(), seed=int(config["study"]["seeds"][0])
    )
    started = time.perf_counter()
    new_predictions = 0
    expected_per_seed = int(config["study"]["predictions_per_seed"])
    for seed_value in config["study"]["seeds"]:
        seed = int(seed_value)
        trials = build_seed_trials(
            model,
            load_source_records(_source_path(config, seed)),
            seed=seed,
            alias_salt=str(config["factorial_design"]["alias_randomization_salt"]),
            corruption_salt=str(
                config["factorial_design"]["corruption_randomization_salt"]
            ),
            token_match_tolerance=int(
                config["factorial_design"]["token_match_tolerance"]
            ),
        )
        design_report = validate_seed_trials(trials, expected_per_seed=expected_per_seed)
        if not design_report["passed"]:
            raise RuntimeError(f"seed {seed} design integrity failed: {design_report}")
        if design_digest(trials) != freeze["design_sha256_by_seed"][str(seed)]:
            raise RuntimeError(f"seed {seed} design differs from preregistration")
        for trial in trials:
            if trial.key in by_key:
                continue
            prediction = model.reason(
                trial.record,
                trial.condition,
                prepared_input=trial.prepared,
            )
            evaluation = model.evaluate_detailed(prediction, trial.record).to_dict()
            row = _prediction_row(trial, prediction, evaluation, freeze)
            _append_jsonl(raw_path, row)
            by_key[trial.key] = row
            new_predictions += 1
            if new_predictions % 25 == 0:
                print(
                    f"PIVOT_EXP_A2 new={new_predictions} total={len(by_key)} "
                    f"elapsed_s={time.perf_counter() - started:.1f}",
                    flush=True,
                )
        observed = sum(int(row["seed"]) == seed for row in by_key.values())
        if observed != expected_per_seed:
            raise RuntimeError(f"seed {seed} count mismatch: {observed} != {expected_per_seed}")
        print(f"PIVOT_EXP_A2 seed={seed} complete predictions={observed}", flush=True)
    rows = list(by_key.values())
    integrity = validate_prediction_rows(rows, config, freeze)
    if not integrity["passed"]:
        raise RuntimeError(f"post-inference integrity failed: {integrity}")
    run = {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A2",
        "status": "inference_complete",
        "prediction_count": len(rows),
        "new_predictions": new_predictions,
        "elapsed_seconds": time.perf_counter() - started,
        "freeze_name": freeze["freeze_name"],
        "freeze_sha256": _sha256(FREEZE_PATH),
        "git": _git_identity(),
        "runtime": runtime_provenance(),
        "model_provenance": model.provenance(),
        "integrity": integrity,
        "adjudication_pending": True,
    }
    _write_json(output / "run.json", run)
    return run


def adjudicate_causal_separation(
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_causal_config(config_path)
    freeze = _verify_freeze(config)
    output = _output_root(config)
    rows = _read_jsonl(output / "predictions.jsonl")
    integrity = validate_prediction_rows(rows, config, freeze)
    power = yaml.safe_load((STUDY_ROOT / "power_analysis.yaml").read_text(encoding="utf-8"))
    analysis = analyze_causal_predictions(rows, config)
    analysis["integrity"] = integrity
    analysis["freeze_name"] = freeze["freeze_name"]
    analysis["model"] = config["model"]["identifier"]
    analysis["revision"] = config["model"]["revision"]
    decision = adjudicate_mechanism(
        analysis,
        config,
        integrity_passed=integrity["passed"],
        power_passed=power["status"] == "PASS",
    )
    analysis["decision"] = decision
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    _promote_results(config, analysis, decision, run, rows, freeze)
    return {"analysis": analysis, "decision": decision}


def _prediction_row(
    trial: CausalTrial,
    prediction: str,
    evaluation: dict[str, Any],
    freeze: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A2",
        "seed": trial.seed,
        "source_scene_id": trial.source_scene_id,
        "trial_id": trial.trial_id,
        "trial_key": trial.key,
        "family": trial.family,
        "condition": trial.condition,
        "task": trial.task,
        "question_type": trial.record.metadata.get("question_type"),
        "hop_depth": int(trial.record.metadata.get("hop_depth", 0)),
        "answer": trial.record.answer,
        "choices": list(trial.record.choices),
        "prediction": prediction,
        "correct": bool(evaluation["correct"]),
        "evaluation": evaluation,
        "factors": trial.factors,
        "input": {
            **trial.prepared.to_dict(),
            "prompt_sha256": hashlib.sha256(trial.prepared.prompt.encode("utf-8")).hexdigest(),
            "image_sha256": trial.record.metadata.get("image_sha256"),
        },
        "metadata": trial.metadata,
        "freeze_name": freeze["freeze_name"],
    }


def _promote_results(
    config: dict[str, Any],
    analysis: dict[str, Any],
    decision: dict[str, Any],
    run: dict[str, Any],
    rows: list[dict[str, Any]],
    freeze: dict[str, Any],
) -> None:
    destination = ROOT / config["study"]["promoted_results"]
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "metrics.json", analysis)
    _write_yaml(destination / "manipulation_check.yaml", analysis["manipulation_check"])
    _write_yaml(destination / "primary_estimands.yaml", analysis["estimands"])
    _write_yaml(destination / "statistical_tests.yaml", analysis["statistical_tests"])
    report = {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A2",
        **decision,
        "freeze_name": freeze["freeze_name"],
        "seeds": analysis["completed_seeds"],
    }
    _write_yaml(destination / "decision_report.yaml", report)
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
    (destination / "decision_report.md").write_text(
        _render_decision_report(analysis, decision), encoding="utf-8", newline="\n"
    )
    _write_figures(destination / "figures", analysis)
    names = (
        "metrics.json",
        "predictions.jsonl.gz",
        "manipulation_check.yaml",
        "primary_estimands.yaml",
        "statistical_tests.yaml",
        "decision_report.yaml",
        "decision_report.md",
        "provenance.yaml",
        "figures/semantic_rescue_manipulation.png",
        "figures/relation_effect_by_semantics.png",
        "figures/format_effect_under_oracle.png",
        "figures/semantic_relation_interaction.png",
        "figures/effect_by_hop_depth.png",
    )
    artifacts = {name: artifact_metadata(destination / name) for name in names}
    _write_yaml(
        destination / "artifact_manifest.yaml",
        {
            "schema_version": 1,
            "study_id": "PIVOT_EXP_A2",
            "outcome": decision["outcome"],
            "prediction_lines": len(rows),
            "freeze_sha256": _sha256(FREEZE_PATH),
            "artifacts": artifacts,
        },
    )


def _render_decision_report(analysis: dict[str, Any], decision: dict[str, Any]) -> str:
    manipulation = analysis["manipulation_check"]
    replication_allowed = str(
        decision["authorization"]["independent_replication_allowed"]
    ).lower()
    lines = [
        "# PIVOT_EXP_A2 Conditional Causal Effects Report",
        "",
        "## Scientific adjudication",
        "",
        f"**{decision['outcome']}**",
        "",
        decision["reason"],
        "",
        "## Semantic rescue validation",
        "",
        f"- Oracle accuracy: {manipulation['oracle']['mean']:.4f}",
        f"- Corrupted accuracy: {manipulation['corrupted']['mean']:.4f}",
        f"- Paired rescue gain: {manipulation['oracle_minus_corrupted']['mean']:.4f}",
        f"- Manipulation passed: {manipulation['passed']}",
        "",
        "## Primary conditional effects",
        "",
        "| Estimand | Mean | 95% CI |",
        "| --- | ---: | --- |",
    ]
    for name, summary in analysis["estimands"].items():
        interval = summary["confidence_interval"]
        lines.append(
            f"| {name} | {summary['mean']:.4f} | "
            f"[{interval['lower']:.4f}, {interval['upper']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Next-stage authorization",
            "",
            "```yaml",
            f"independent_replication_allowed: {replication_allowed}",
            "model_development_allowed: false",
            "paper_writing_allowed: false",
            "```",
            "",
            "This LLaVA-only result is not a general claim about modern VLMs.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_figures(path: Path, analysis: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    path.mkdir(parents=True, exist_ok=True)
    manipulation = analysis["manipulation_check"]
    _bar_figure(
        path / "semantic_rescue_manipulation.png",
        ["Oracle", "Corrupted"],
        [manipulation["oracle"]["mean"], manipulation["corrupted"]["mean"]],
        "Semantic rescue manipulation check",
        "Accuracy",
    )
    estimands = analysis["estimands"]
    _bar_figure(
        path / "relation_effect_by_semantics.png",
        ["Oracle relation effect", "Semantic rescue"],
        [
            estimands["E2_relation_under_oracle"]["mean"],
            estimands["E1_semantic_rescue"]["mean"],
        ],
        "Controlled relation and semantic effects",
        "Paired accuracy difference",
    )
    _bar_figure(
        path / "format_effect_under_oracle.png",
        ["JSON - triples"],
        [estimands["E3_json_minus_triples"]["mean"]],
        "Format effect under oracle semantics",
        "Paired accuracy difference",
    )
    _bar_figure(
        path / "semantic_relation_interaction.png",
        ["S x R", "S x F", "S x R x F"],
        [
            estimands["E4_semantic_x_relation"]["mean"],
            estimands["E5_semantic_x_format"]["mean"],
            estimands["E6_three_way"]["mean"],
        ],
        "Registered interactions",
        "Difference in differences",
    )
    figure, axis = plt.subplots(figsize=(7, 4))
    hops = ["1", "2", "3", "4"]
    for estimand, label in (
        ("E1_semantic_rescue", "Semantic rescue"),
        ("E2_relation_under_oracle", "Relation under oracle"),
        ("E3_json_minus_triples", "JSON - triples"),
    ):
        axis.plot(
            [int(hop) for hop in hops],
            [analysis["hop_depth"][hop][estimand]["mean"] for hop in hops],
            marker="o",
            label=label,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Hop depth")
    axis.set_ylabel("Paired accuracy difference")
    axis.set_title("Registered effects by hop depth")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path / "effect_by_hop_depth.png", dpi=160)
    plt.close(figure)


def _bar_figure(
    path: Path, labels: list[str], values: list[float], title: str, ylabel: str
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(labels, values, color="#4472C4")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _verify_freeze(config: dict[str, Any]) -> dict[str, Any]:
    if not FREEZE_PATH.is_file():
        raise FileNotFoundError("PIVOT_EXP_A2 freeze manifest is missing; run preflight first")
    freeze = yaml.safe_load(FREEZE_PATH.read_text(encoding="utf-8"))
    if not isinstance(freeze, dict) or freeze.get("passed") is not True:
        raise ValueError("PIVOT_EXP_A2 freeze manifest is not passing")
    if not locked_files_match(ROOT, freeze):
        raise ValueError("PIVOT_EXP_A2 locked protocol/code drift detected")
    if not source_files_match(ROOT, freeze):
        raise ValueError("PIVOT_EXP_A2 source dataset drift detected")
    parent = validate_parent_freeze(ROOT, config["study"]["parent_freeze"])
    if not parent["passed"]:
        raise ValueError("PIVOT_EXP_A parent freeze no longer validates")
    return freeze


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "study",
        "model",
        "factorial_design",
        "semantic_scaffold",
        "relation_intervention",
        "manipulation_check",
        "statistics",
        "prohibitions",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"causal-separation config missing keys: {missing}")
    if int(config["schema_version"]) != 1:
        raise ValueError("causal-separation schema_version must be 1")
    if len(config["study"]["seeds"]) != 5 or len(set(config["study"]["seeds"])) != 5:
        raise ValueError("PIVOT_EXP_A2 requires five unique frozen seeds")
    if int(config["study"]["primary_scenes_per_seed"]) != 90:
        raise ValueError("PIVOT_EXP_A2 primary scene inventory is frozen at 90 per seed")
    if int(config["study"]["predictions_per_seed"]) != 1060:
        raise ValueError("PIVOT_EXP_A2 requires 1060 predictions per seed")
    if config["model"]["training_enabled"] is not False:
        raise ValueError("PIVOT_EXP_A2 prohibits training")
    if not all(bool(value) for value in config["prohibitions"].values()):
        raise ValueError("all PIVOT_EXP_A2 prohibitions must remain enabled")


def _source_path(config: dict[str, Any], seed: int) -> Path:
    return ROOT / str(config["study"]["source_dataset_template"]).format(seed=seed)


def _output_root(config: dict[str, Any]) -> Path:
    return ROOT / config["study"]["output_root"]


def _git_identity() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, check=False, text=True
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain=v1", "--untracked-files=all")),
    }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


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
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_yaml(path: Path, payload: Any) -> None:
    serializable = json.loads(json.dumps(payload, allow_nan=False, default=str))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(serializable, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )


__all__ = [
    "DEFAULT_CONFIG",
    "FREEZE_PATH",
    "adjudicate_causal_separation",
    "load_causal_config",
    "preregister_causal_separation",
    "run_causal_separation",
    "validate_causal_separation",
]
