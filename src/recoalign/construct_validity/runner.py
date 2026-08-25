"""Single preregister/preflight/run/adjudicate lifecycle for PIVOT_EXP_A3."""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from recoalign.models.vlm.base import PreparedInput
from recoalign.models.vlm.registry import ModelRegistry
from recoalign.synthetic_world import GeneratorConfig, SyntheticWorldGenerator

from .answer_contract import REGISTERED_CHOICE_IDS, parse_final_choice, validate_parser_contract
from .choice_scoring import (
    LlavaConditionalLikelihoodBackend,
    score_choices,
    validate_tokenizer_contract,
)
from .decision import adjudicate_construct_validity
from .integrity import (
    a2_scene_ids,
    artifact_metadata,
    audit_legacy_parse_failures,
    validate_a2_frozen_assets,
    validate_prediction_integrity,
    validate_trial_inventory,
)
from .scaffold_comprehension import build_scaffold_pair
from .statistics import analyze_construct_predictions
from .trial_builder import build_validation_trials
from .visual_legend import build_m2_scene_assets, create_neutral_image

ROOT = Path(__file__).resolve().parents[3]
STUDY_ROOT = ROOT / "research/construct_validity/PIVOT_EXP_A3"
DEFAULT_CONFIG = STUDY_ROOT / "config.yaml"
FREEZE_PATH = STUDY_ROOT / "freeze_manifest.yaml"
VALIDATION_ROOT = STUDY_ROOT / "validation"
TRIAL_INVENTORY = VALIDATION_ROOT / "trial_inventory.jsonl.gz"

_LOCKED_PROTOCOL_FILES = (
    "research/decisions/PIVOT_EXP_A2_continuation_scope.yaml",
    "research/construct_validity/PIVOT_EXP_A3/research_question.md",
    "research/construct_validity/PIVOT_EXP_A3/hypothesis_registry.yaml",
    "research/construct_validity/PIVOT_EXP_A3/preregistration.md",
    "research/construct_validity/PIVOT_EXP_A3/power_analysis.yaml",
    "research/construct_validity/PIVOT_EXP_A3/config.yaml",
    "research/construct_validity/PIVOT_EXP_A3/decision_policy.yaml",
    "research/construct_validity/PIVOT_EXP_A3/answer_contract/answer-contract-v2.yaml",
    "research/construct_validity/PIVOT_EXP_A3/answer_contract/tokenizer_contract.yaml",
    "research/construct_validity/PIVOT_EXP_A3/semantic_manipulations/M0_historical_anchor.md",
    "research/construct_validity/PIVOT_EXP_A3/semantic_manipulations/M1_canonical_entity_table.md",
    "research/construct_validity/PIVOT_EXP_A3/semantic_manipulations/M2_visual_object_legend.md",
    "research/construct_validity/PIVOT_EXP_A3/protocols/answer_contract_protocol.md",
    "research/construct_validity/PIVOT_EXP_A3/protocols/sampling_and_trial_protocol.md",
    "research/construct_validity/PIVOT_EXP_A3/protocols/statistical_protocol.md",
    "research/construct_validity/PIVOT_EXP_A3/validation/answer_contract_preflight.yaml",
    "research/construct_validity/PIVOT_EXP_A3/validation/legacy_parse_failure_inventory.jsonl",
    "research/construct_validity/PIVOT_EXP_A3/validation/legacy_parse_failure_summary.yaml",
    "research/construct_validity/PIVOT_EXP_A3/validation/legacy_parse_failure_audit.md",
    "src/recoalign/construct_validity/answer_contract.py",
    "src/recoalign/construct_validity/choice_scoring.py",
    "src/recoalign/construct_validity/scaffold_comprehension.py",
    "src/recoalign/construct_validity/visual_legend.py",
    "src/recoalign/construct_validity/trial_builder.py",
    "src/recoalign/construct_validity/statistics.py",
    "src/recoalign/construct_validity/decision.py",
    "src/recoalign/construct_validity/integrity.py",
    "src/recoalign/construct_validity/runner.py",
)


def load_construct_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("construct-validity config must be a mapping")
    _validate_config(payload)
    return payload


def validate_answer_contract(
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_construct_config(config_path)
    parser_report = validate_parser_contract()
    tokenizer = _load_frozen_tokenizer(config)
    tokenizer_report = validate_tokenizer_contract(tokenizer)
    report = {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A3",
        "weights_loaded": False,
        "inference_started": False,
        "parser": parser_report,
        "tokenizer": tokenizer_report,
        "passed": bool(parser_report["passed"] and tokenizer_report["passed"]),
    }
    _write_yaml(VALIDATION_ROOT / "answer_contract_preflight.yaml", report)
    _write_yaml(STUDY_ROOT / "answer_contract/tokenizer_contract.yaml", tokenizer_report)
    return report


def preregister_construct_validity(
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    load_construct_config(config_path)
    power = yaml.safe_load((STUDY_ROOT / "power_analysis.yaml").read_text(encoding="utf-8"))
    if power.get("status") != "PASS" or power.get("computed_before_inference") is not True:
        raise ValueError("PIVOT_EXP_A3 power analysis is not a passing pre-inference record")
    boundary = validate_a2_frozen_assets(ROOT)
    if not boundary["passed"]:
        raise ValueError("PIVOT_EXP_A2 frozen boundary failed")
    audit = audit_legacy_parse_failures(ROOT, VALIDATION_ROOT.relative_to(ROOT))
    contract = validate_answer_contract(config_path)
    if not contract["passed"]:
        raise ValueError("PIVOT_EXP_A3 answer-contract preflight failed")
    neutral = create_neutral_image(STUDY_ROOT / "semantic_manipulations/neutral.png")
    locked = {relative: _sha256(ROOT / relative) for relative in _LOCKED_PROTOCOL_FILES}
    locked["research/construct_validity/PIVOT_EXP_A3/semantic_manipulations/neutral.png"] = (
        neutral["sha256"]
    )
    report = {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A3",
        "status": "preregistered_preflight_pending",
        "registered_at": "2026-08-23",
        "inference_started": False,
        "weights_loaded": False,
        "protocol_changes_allowed_after_inference": False,
        "parent_boundary": boundary,
        "legacy_audit": {
            "selected_count": audit["selection"]["selected_count"],
            "rescored": False,
            "source_sha256": audit["source_predictions_sha256"],
        },
        "power": {"status": power["status"], "sha256": _sha256(STUDY_ROOT / "power_analysis.yaml")},
        "answer_contract": {
            "passed": contract["passed"],
            "parser_passed": contract["parser"]["passed"],
            "tokenizer_passed": contract["tokenizer"]["passed"],
        },
        "locked_protocol_files": locked,
        "preflight": {"passed": False, "validation_inventory_frozen": False},
    }
    _write_yaml(FREEZE_PATH, report)
    _write_yaml(
        VALIDATION_ROOT / "preregistration_report.yaml",
        {
            "study_id": "PIVOT_EXP_A3",
            "status": report["status"],
            "passed": True,
            "weights_loaded": False,
            "inference_started": False,
            "power": power["status"],
            "A2_boundary": boundary["passed"],
            "answer_contract": contract["passed"],
        },
    )
    return report


def validate_construct_validity(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    preflight_only: bool = True,
) -> dict[str, Any]:
    del preflight_only
    config = load_construct_config(config_path)
    freeze = _load_and_verify_protocol_freeze()
    output_root = ROOT / config["study"]["output_root"]
    inventory_root = output_root / "inventory"
    neutral = STUDY_ROOT / "semantic_manipulations/neutral.png"
    development = _materialize_inventory(
        seed=int(config["study"]["development_seed"]),
        selected_count=int(config["study"]["development_scenes"]),
        candidate_count=96,
        output_dir=inventory_root / "development",
    )
    development_ids = {record.scene_id for record in development}
    validation_records: dict[int, list[Any]] = {}
    all_trial_rows = []
    legend_reports = []
    asset_rows = []
    development_mapping: dict[tuple[str, str, str], str] = {}
    development_legend_reports = []
    development_seed = int(config["study"]["development_seed"])
    for record in development:
        mapping, legend_report = build_m2_scene_assets(
            record,
            build_scaffold_pair(record, "M2", seed=development_seed),
            neutral_image=neutral,
            output_dir=inventory_root / "m2_development" / record.scene_id,
        )
        development_mapping.update(mapping)
        development_legend_reports.append(legend_report)
        asset_rows.extend(legend_report["tiles"])
        for condition in legend_report["conditions"].values():
            asset_rows.append(condition["legend"])
            asset_rows.extend(condition["contexts"].values())
    development_trials = build_validation_trials(
        development,
        seed=development_seed,
        neutral_image=neutral,
        m2_images=development_mapping,
    )
    development_preflight = {
        "passed": (
            len(development_trials) == len(development) * 72
            and len({trial.key for trial in development_trials}) == len(development_trials)
            and all(bool(report["passed"]) for report in development_legend_reports)
        ),
        "scene_count": len(development),
        "trial_schema_count": len(development_trials),
        "accuracy_computed": False,
        "used_for_prompt_selection": False,
    }
    for seed_value in config["study"]["validation_seeds"]:
        seed = int(seed_value)
        records = _materialize_inventory(
            seed=seed,
            selected_count=int(config["study"]["validation_scenes_per_seed"]),
            candidate_count=int(config["study"]["generated_candidates_per_validation_seed"]),
            output_dir=inventory_root / "validation" / str(seed),
        )
        validation_records[seed] = records
        m2_mapping: dict[tuple[str, str, str], str] = {}
        for record in records:
            mapping, legend_report = build_m2_scene_assets(
                record,
                build_scaffold_pair(record, "M2", seed=seed),
                neutral_image=neutral,
                output_dir=inventory_root / "m2" / str(seed) / record.scene_id,
            )
            m2_mapping.update(mapping)
            legend_reports.append(legend_report)
            asset_rows.extend(legend_report["tiles"])
            for condition in legend_report["conditions"].values():
                asset_rows.append(condition["legend"])
                asset_rows.extend(condition["contexts"].values())
        trials = build_validation_trials(
            records,
            seed=seed,
            neutral_image=neutral,
            m2_images=m2_mapping,
        )
        all_trial_rows.extend(trial.to_dict() for trial in trials)
    trial_integrity = validate_trial_inventory(
        all_trial_rows,
        config,
        frozen_a2_scene_ids=a2_scene_ids(ROOT),
        development_scene_ids=development_ids,
    )
    legend_integrity = {
        "passed": bool(legend_reports) and all(bool(report["passed"]) for report in legend_reports),
        "scene_count": len(legend_reports),
        "oracle_corrupted_dimensions_equal": all(
            bool(report["assertions"]["legend_dimensions_equal"])
            for report in legend_reports
        ),
        "absolute_coordinates_preserved": False,
    }
    prompt_tokens = _validate_prompt_token_matching(all_trial_rows, config)
    contract = validate_answer_contract(config_path)
    neutral_integrity = {
        "passed": all(
            Path(str(row["image"])).resolve() == neutral.resolve()
            for row in all_trial_rows
            if row["image_context"] == "neutral_image"
            and row["manipulation"] in {"M0", "M1"}
        ),
        "path": neutral.relative_to(ROOT).as_posix(),
        "sha256": _sha256(neutral),
        "single_frozen_neutral_source": True,
    }
    design_passed = bool(
        trial_integrity["passed"]
        and legend_integrity["passed"]
        and development_preflight["passed"]
        and prompt_tokens["passed"]
        and neutral_integrity["passed"]
        and contract["passed"]
        and validate_a2_frozen_assets(ROOT)["passed"]
    )
    _write_jsonl_gz(TRIAL_INVENTORY, all_trial_rows)
    asset_manifest_path = VALIDATION_ROOT / "visual_asset_manifest.jsonl"
    _write_jsonl(asset_manifest_path, sorted(asset_rows, key=lambda row: str(row["path"])))
    inventory_files = {
        TRIAL_INVENTORY.relative_to(ROOT).as_posix(): artifact_metadata(TRIAL_INVENTORY),
        asset_manifest_path.relative_to(ROOT).as_posix(): artifact_metadata(asset_manifest_path),
    }
    for seed, records in validation_records.items():
        dataset = inventory_root / "validation" / str(seed) / "dataset.jsonl"
        inventory_files[dataset.relative_to(ROOT).as_posix()] = artifact_metadata(dataset)
        if len(records) != int(config["study"]["validation_scenes_per_seed"]):
            design_passed = False
    development_dataset = inventory_root / "development" / "dataset.jsonl"
    inventory_files[development_dataset.relative_to(ROOT).as_posix()] = artifact_metadata(
        development_dataset
    )
    inventory_manifest = {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A3",
        "passed": design_passed,
        "files": inventory_files,
        "validation_scene_ids_by_seed": {
            seed: [record.scene_id for record in records]
            for seed, records in sorted(validation_records.items())
        },
        "development_scene_ids": sorted(development_ids),
        "development_accuracy_computed": False,
        "visual_asset_count": len(asset_rows),
    }
    _write_yaml(VALIDATION_ROOT / "inventory_manifest.yaml", inventory_manifest)
    runtime_readiness = _runtime_readiness(config)
    _write_yaml(VALIDATION_ROOT / "runtime_readiness.yaml", runtime_readiness)
    passed = bool(design_passed and runtime_readiness["runtime_ready"])
    status = (
        "frozen_preinference"
        if passed
        else "preflight_blocked_runtime"
        if design_passed
        else "preflight_failed"
    )
    report = {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A3",
        "status": status,
        "passed": passed,
        "design_preflight_passed": design_passed,
        "execution_runtime_ready": bool(runtime_readiness["runtime_ready"]),
        "preflight_only": True,
        "weights_loaded": False,
        "inference_started": False,
        "trial_integrity": trial_integrity,
        "legend_integrity": legend_integrity,
        "development_preflight": development_preflight,
        "prompt_token_matching": prompt_tokens,
        "neutral_image_integrity": neutral_integrity,
        "answer_contract": contract,
        "runtime_readiness": runtime_readiness,
        "inventory_manifest_sha256": _sha256(VALIDATION_ROOT / "inventory_manifest.yaml"),
        "trial_inventory_sha256": _sha256(TRIAL_INVENTORY),
    }
    _write_yaml(VALIDATION_ROOT / "preflight_report.yaml", report)
    freeze["status"] = report["status"]
    freeze["preflight"] = {
        "passed": passed,
        "design_preflight_passed": design_passed,
        "validation_inventory_frozen": design_passed,
        "execution_runtime_ready": bool(runtime_readiness["runtime_ready"]),
        "report_sha256": _sha256(VALIDATION_ROOT / "preflight_report.yaml"),
        "inventory_manifest_sha256": report["inventory_manifest_sha256"],
        "trial_inventory_sha256": report["trial_inventory_sha256"],
        "trial_count": len(all_trial_rows),
        "visual_asset_count": len(asset_rows),
    }
    _write_yaml(FREEZE_PATH, freeze)
    return report


def run_construct_validity(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    model_name: str = "llava_1_5_7b",
) -> dict[str, Any]:
    config = load_construct_config(config_path)
    if model_name != config["model"]["registry_name"]:
        raise ValueError("construct-validity run must use the preregistered LLaVA model")
    _verify_execution_freeze()
    execution_contract = validate_answer_contract(config_path)
    if not execution_contract["passed"]:
        raise ValueError("answer contract drifted at the execution boundary")
    trial_rows = _read_jsonl_gz(TRIAL_INVENTORY)
    output = ROOT / config["study"]["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.jsonl"
    existing = _read_jsonl(predictions_path)
    predictions = {str(row["trial_key"]): row for row in existing}
    if len(predictions) != len(existing):
        raise ValueError("construct-validity predictions contain duplicate trial keys")
    registry = ModelRegistry(ROOT)
    definition = registry.definition(model_name)
    model = registry.get_or_create(
        definition.experiment_model_config(),
        seed=int(config["study"]["validation_seeds"][0]),
    )
    likelihood_backend = LlavaConditionalLikelihoodBackend(model)
    started = time.perf_counter()
    added = 0
    for trial in trial_rows:
        if trial["trial_key"] in predictions:
            continue
        prepared = _prepared_from_trial(trial)
        if trial["response_method"] == "forced_choice":
            scored = score_choices(prepared, REGISTERED_CHOICE_IDS, backend=likelihood_backend)
            prediction_choice = scored.predicted_choice_id
            result = {
                "raw_output": prediction_choice,
                "prediction_choice_id": prediction_choice,
                "valid_measurement": scored.valid_measurement,
                "parsed": True,
                "parse_failure_reason": None,
                "choice_scores": scored.log_likelihoods,
                "scoring_method": scored.method,
                "tied_maximum": scored.tied_maximum,
            }
        else:
            raw = model.generate(
                prepared.prompt,
                image=prepared.image,
                temperature=0.0,
                do_sample=False,
                max_new_tokens=16,
                num_beams=1,
            )
            parsed = parse_final_choice(raw)
            prediction_choice = parsed.choice_id
            result = {
                "raw_output": raw,
                "prediction_choice_id": prediction_choice,
                "valid_measurement": False,
                "parsed": parsed.valid,
                "parse_failure_reason": parsed.failure_reason,
                "choice_scores": None,
                "scoring_method": "frozen_free_generation_answer-contract-v2",
                "tied_maximum": False,
            }
        row = {
            **trial,
            **result,
            "correct": prediction_choice == trial["correct_choice_id"],
            "freeze_sha256": _sha256(FREEZE_PATH),
        }
        _append_jsonl(predictions_path, row)
        predictions[str(trial["trial_key"])] = row
        added += 1
        if added % 25 == 0:
            print(
                f"PIVOT_EXP_A3 new={added} total={len(predictions)} "
                f"elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )
    rows = list(predictions.values())
    integrity = validate_prediction_integrity(rows, trial_rows, config)
    if not integrity["passed"]:
        raise RuntimeError(f"post-inference construct-validity integrity failed: {integrity}")
    run = {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A3",
        "status": "inference_complete_adjudication_pending",
        "prediction_count": len(rows),
        "new_predictions": added,
        "elapsed_seconds": time.perf_counter() - started,
        "integrity": integrity,
        "model_provenance": model.provenance(),
        "adjudication_pending": True,
    }
    _write_json(output / "run.json", run)
    return run


def adjudicate_construct_validity_run(
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_construct_config(config_path)
    freeze = _verify_execution_freeze(allow_inference_complete=True)
    output = ROOT / config["study"]["output_root"]
    trial_rows = _read_jsonl_gz(TRIAL_INVENTORY)
    rows = _read_jsonl(output / "predictions.jsonl")
    integrity = validate_prediction_integrity(rows, trial_rows, config)
    analysis = analyze_construct_predictions(rows, config)
    decision = adjudicate_construct_validity(analysis, integrity_passed=integrity["passed"])
    _promote_results(config, rows, analysis, decision, integrity, freeze)
    return {"analysis": analysis, "decision": decision}


def _materialize_inventory(
    *, seed: int, selected_count: int, candidate_count: int, output_dir: Path
) -> list[Any]:
    generator = SyntheticWorldGenerator(
        GeneratorConfig(seed=seed, image_size=192, write_images=True, style="flat")
    )
    candidates = generator.generate(candidate_count, seed=seed, split="construct_validity")
    selected = sorted(
        (record for record in candidates if len(record.objects) == 4),
        key=lambda record: record.scene_id,
    )[:selected_count]
    if len(selected) != selected_count:
        raise ValueError(f"seed {seed}: insufficient four-entity candidates")
    materialized = generator.materialize(selected, output_dir)
    return [
        replace(record, metadata={**record.metadata, "construct_validity_selected": True})
        for record in materialized
    ]


def _validate_prompt_token_matching(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    tokenizer = _load_frozen_tokenizer(config)
    choice_tokens = {
        choice: tokenizer(choice, add_special_tokens=False)["input_ids"]
        for choice in REGISTERED_CHOICE_IDS
    }
    groups: dict[tuple[Any, ...], dict[str, int]] = defaultdict(dict)
    boundary_failures = []
    for row in rows:
        if row["response_method"] != "forced_choice":
            continue
        key = (
            int(row["seed"]),
            str(row["scene_id"]),
            str(row["construct"]),
            str(row["manipulation"]),
            str(row["image_context"]),
            str(row["task"]),
        )
        prompt = str(row["prompt"])
        prefix_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        groups[key][str(row["evidence_truth"])] = len(prefix_ids)
        for choice, expected_suffix in choice_tokens.items():
            combined = tokenizer(prompt + choice, add_special_tokens=True)["input_ids"]
            if combined[: len(prefix_ids)] != prefix_ids or combined[len(prefix_ids) :] != (
                expected_suffix
            ):
                if len(boundary_failures) < 20:
                    boundary_failures.append(
                        {"trial_key": row["trial_key"], "choice_id": choice}
                    )
    deltas = [abs(values["oracle"] - values["corrupted"]) for values in groups.values()]
    tolerance = 0
    return {
        "passed": bool(deltas) and max(deltas) <= tolerance and not boundary_failures,
        "tolerance": tolerance,
        "maximum_delta": max(deltas) if deltas else None,
        "paired_groups": len(groups),
        "full_prompt_candidate_boundary_passed": not boundary_failures,
        "boundary_failures": boundary_failures,
    }


def _prepared_from_trial(trial: dict[str, Any]) -> PreparedInput:
    return PreparedInput(
        prompt=str(trial["prompt"]),
        image=str(trial["image"]),
        condition=(
            f"{trial['manipulation']}_{trial['evidence_truth']}_{trial['image_context']}"
        ),
        setting="construct_validity",
        input_tokens=0,
        evidence_tokens=0,
        semantic_units=4,
        relation_count=0,
        semantic_facts_sha256=str(trial["scaffold_sha256"]),
        serialization=str(trial["manipulation"]),
        padding_units=0,
        token_match_delta=0,
        prompt_protocol_id="pivot-exp-a3-answer-contract-v2",
        prompt_protocol_sha256=None,
    )


def _promote_results(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    analysis: dict[str, Any],
    decision: dict[str, Any],
    integrity: dict[str, Any],
    freeze: dict[str, Any],
) -> None:
    results = STUDY_ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    _write_json(results / "metrics.json", {**analysis, "decision": decision})
    _write_jsonl_gz(results / "predictions.jsonl.gz", rows)
    _write_yaml(
        results / "answer_contract_report.yaml",
        {
            "global_measurement_validity": analysis["gate_A"],
            "candidates": {
                name: value["gate_A"] for name, value in analysis["candidates"].items()
            },
        },
    )
    _write_yaml(
        results / "parse_integrity_report.yaml",
        {name: value["gate_B"] for name, value in analysis["candidates"].items()},
    )
    _write_yaml(
        results / "scaffold_comprehension.yaml",
        {name: value["gate_C"] for name, value in analysis["candidates"].items()},
    )
    _write_yaml(
        results / "semantic_sufficiency.yaml",
        {name: value["gate_D"] for name, value in analysis["candidates"].items()},
    )
    _write_yaml(
        results / "image_interference.yaml",
        {name: value["gate_F"] for name, value in analysis["candidates"].items()},
    )
    _write_yaml(
        results / "task_specific_results.yaml",
        {
            name: {
                "C": value["gate_C"]["by_task"],
                "D": value["gate_D"]["by_task"],
                "E": value["gate_E"]["by_task"],
                "F": value["gate_F"]["by_task"],
            }
            for name, value in analysis["candidates"].items()
        },
    )
    _write_yaml(
        results / "statistical_tests.yaml",
        {
            name: {
                "gate_E_Holm": value["gate_E"]["holm_family"],
                "gate_F_Holm_TOST": value["gate_F"]["holm_family"],
            }
            for name, value in analysis["candidates"].items()
        },
    )
    _write_yaml(results / "decision_report.yaml", decision)
    (results / "decision_report.md").write_text(
        _decision_markdown(decision), encoding="utf-8", newline="\n"
    )
    _write_yaml(
        results / "provenance.yaml",
        {
            "study_id": "PIVOT_EXP_A3",
            "model": config["model"],
            "freeze_sha256": _sha256(FREEZE_PATH),
            "trial_inventory_sha256": freeze["preflight"]["trial_inventory_sha256"],
            "integrity": integrity,
        },
    )
    _build_figures(rows, analysis, results / "figures")
    artifacts = {}
    for path in sorted(results.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.yaml":
            artifacts[path.relative_to(results).as_posix()] = artifact_metadata(path)
    _write_yaml(
        results / "artifact_manifest.yaml",
        {
            "schema_version": 1,
            "study_id": "PIVOT_EXP_A3",
            "outcome": decision["outcome"],
            "prediction_lines": len(rows),
            "artifacts": artifacts,
        },
    )


def _build_figures(
    rows: list[dict[str, Any]], analysis: dict[str, Any], output_dir: Path
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = ["shape", "color", "object_identity", "entity_attribute_binding"]

    def save_bars(path: Path, labels: list[str], values: list[float], title: str) -> None:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar(range(len(values)), values, color="#4472C4")
        axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        axis.set_ylim(0.0, 1.05)
        axis.set_title(title)
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)

    save_bars(
        output_dir / "task_specific_oracle_sufficiency.png",
        tasks,
        [analysis["candidates"]["M1"]["gate_D"]["by_task"][task]["mean"] for task in tasks],
        "M1 oracle scene-truth sufficiency by task",
    )
    for filename, manipulation, gate, title in (
        ("oracle_vs_corrupted_by_task.png", "M1", "gate_E", "M1 oracle-corrupted gain"),
        ("text_table_vs_visual_legend.png", "M2", "gate_D", "M2 oracle sufficiency"),
        ("neutral_vs_original.png", "M1", "gate_F", "M1 neutral-original difference"),
        ("confidence_intervals_against_gates.png", "M2", "gate_E", "M2 separation"),
    ):
        values = []
        for task in tasks:
            summary = analysis["candidates"][manipulation][gate]["by_task"][task]
            values.append(float(summary.get("mean", 1.0 if summary.get("passed") else 0.0)))
        save_bars(output_dir / filename, tasks, values, title)
    methods = Counter((row["response_method"], bool(row.get("parsed", True))) for row in rows)
    forced_rate = methods[("forced_choice", True)] / max(
        1, sum(value for (method, _), value in methods.items() if method == "forced_choice")
    )
    free_rate = methods[("free_generation", True)] / max(
        1, sum(value for (method, _), value in methods.items() if method == "free_generation")
    )
    save_bars(
        output_dir / "forced_choice_vs_free_generation_validity.png",
        ["forced choice", "free generation"],
        [forced_rate, free_rate],
        "Answer measurement validity",
    )
    failures = Counter(
        str(row.get("parse_failure_reason") or "valid")
        for row in rows
        if row["response_method"] == "free_generation"
    )
    save_bars(
        output_dir / "parse_failure_taxonomy.png",
        list(failures),
        [value / sum(failures.values()) for value in failures.values()],
        "Free-generation parse taxonomy",
    )
    seed_values = defaultdict(list)
    for row in rows:
        if (
            row["response_method"] == "forced_choice"
            and row["manipulation"] == "M1"
            and row["construct"] == "CV3"
            and row["evidence_truth"] == "oracle"
        ):
            seed_values[int(row["seed"])].append(float(bool(row["correct"])))
    save_bars(
        output_dir / "seed_level_replication.png",
        [str(seed) for seed in sorted(seed_values)],
        [sum(seed_values[seed]) / len(seed_values[seed]) for seed in sorted(seed_values)],
        "M1 oracle accuracy by seed",
    )


def _decision_markdown(decision: dict[str, Any]) -> str:
    authorization = yaml.safe_dump(decision["authorization"], sort_keys=False).rstrip()
    return f"""# PIVOT_EXP_A3 Decision

## Formal outcome

`{decision['outcome']}`

Selected manipulation: `{decision['selected_manipulation']}`

{decision['reason']}

## Next-stage authorization

```yaml
{authorization}
```
"""


def _load_and_verify_protocol_freeze() -> dict[str, Any]:
    if not FREEZE_PATH.is_file():
        raise FileNotFoundError("PIVOT_EXP_A3 is not preregistered")
    freeze = yaml.safe_load(FREEZE_PATH.read_text(encoding="utf-8"))
    if not isinstance(freeze, dict) or freeze.get("inference_started") is not False:
        raise ValueError("PIVOT_EXP_A3 protocol freeze is not in a pre-inference state")
    mismatches = [
        relative
        for relative, digest in freeze["locked_protocol_files"].items()
        if _sha256(ROOT / relative) != digest
    ]
    if mismatches:
        raise ValueError(f"PIVOT_EXP_A3 locked protocol drift: {mismatches}")
    if not validate_a2_frozen_assets(ROOT)["passed"]:
        raise ValueError("PIVOT_EXP_A2 frozen boundary drift")
    return freeze


def _verify_execution_freeze(*, allow_inference_complete: bool = False) -> dict[str, Any]:
    if not FREEZE_PATH.is_file():
        raise FileNotFoundError("PIVOT_EXP_A3 freeze manifest is missing")
    freeze = yaml.safe_load(FREEZE_PATH.read_text(encoding="utf-8"))
    del allow_inference_complete
    allowed = {"frozen_preinference"}
    if freeze.get("status") not in allowed or freeze.get("preflight", {}).get("passed") is not True:
        raise ValueError("PIVOT_EXP_A3 execution barrier is not passing")
    if _sha256(TRIAL_INVENTORY) != freeze["preflight"]["trial_inventory_sha256"]:
        raise ValueError("PIVOT_EXP_A3 trial inventory hash drift")
    if not _verify_inventory_assets(freeze):
        raise ValueError("PIVOT_EXP_A3 frozen inventory or visual asset drift")
    for relative, digest in freeze["locked_protocol_files"].items():
        if _sha256(ROOT / relative) != digest:
            raise ValueError(f"PIVOT_EXP_A3 locked protocol drift: {relative}")
    if not validate_a2_frozen_assets(ROOT)["passed"]:
        raise ValueError("PIVOT_EXP_A2 frozen boundary drift")
    return freeze


def _verify_inventory_assets(freeze: dict[str, Any]) -> bool:
    manifest_path = VALIDATION_ROOT / "inventory_manifest.yaml"
    if _sha256(manifest_path) != freeze["preflight"]["inventory_manifest_sha256"]:
        return False
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    for relative, metadata in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file():
            return False
        if path.stat().st_size != int(metadata["bytes"]) or _sha256(path) != metadata["sha256"]:
            return False
    visual_manifest = VALIDATION_ROOT / "visual_asset_manifest.jsonl"
    for row in _read_jsonl(visual_manifest):
        path = Path(row["path"])
        if not path.is_file():
            return False
        if path.stat().st_size != int(row["bytes"]) or _sha256(path) != row["sha256"]:
            return False
    return True


def _load_frozen_tokenizer(config: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        ROOT / config["model"]["tokenizer_path"],
        local_files_only=True,
        use_fast=False,
    )


def _runtime_readiness(config: dict[str, Any]) -> dict[str, Any]:
    registry = ModelRegistry(ROOT)
    definition = registry.definition(str(config["model"]["registry_name"]))
    model = registry.get_or_create(
        definition.experiment_model_config(),
        seed=int(config["study"]["validation_seeds"][0]),
    )
    report = model.dry_run()
    if report.get("weights_loaded") is not False:
        raise RuntimeError("runtime readiness preflight must not load model weights")
    return report


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "study",
        "model",
        "answer_contract",
        "semantic_manipulations",
        "design",
        "statistics",
        "gates",
        "execution_barrier",
        "prohibitions",
    }
    if missing := sorted(required - set(config)):
        raise ValueError(f"construct-validity config missing keys: {missing}")
    seeds = [int(value) for value in config["study"]["validation_seeds"]]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("PIVOT_EXP_A3 requires five unique validation seeds")
    if int(config["study"]["validation_scenes_per_seed"]) != 100:
        raise ValueError("PIVOT_EXP_A3 is powered and frozen at 100 scenes per seed")
    if tuple(str(value) for value in config["answer_contract"]["option_ids"]) != (
        "1",
        "2",
        "3",
        "4",
    ):
        raise ValueError("PIVOT_EXP_A3 option IDs must be 1/2/3/4")
    if config["answer_contract"]["primary_method"] != (
        "conditional_log_likelihood_single_token_argmax_v1"
    ):
        raise ValueError("primary answer method drifted from preregistration")
    if config["model"]["training_enabled"] is not False:
        raise ValueError("PIVOT_EXP_A3 prohibits model training")
    if not all(bool(value) for value in config["prohibitions"].values()):
        raise ValueError("all PIVOT_EXP_A3 prohibitions must remain enabled")


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in rows:
                compressed.write(
                    (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
                )


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
    "TRIAL_INVENTORY",
    "adjudicate_construct_validity_run",
    "load_construct_config",
    "preregister_construct_validity",
    "run_construct_validity",
    "validate_answer_contract",
    "validate_construct_validity",
]
