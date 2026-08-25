"""Pre-inference freeze, resumable execution, and reporting for PIVOT_EXP_A3P."""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from recoalign.models.vlm.base import PreparedInput
from recoalign.models.vlm.registry import ModelRegistry

from .decision import adjudicate_primary_construct_validity as decide
from .integrity import (
    create_a3r_final_manifest,
    git_execution_boundary,
    payload_sha256,
    scorer_boundary_report,
    stable_environment_payload,
    validate_prediction_integrity,
    verify_freeze,
    verify_parent_freezes,
    write_json_atomic,
    write_yaml_atomic,
)
from .inventory import (
    DEVELOPMENT_INVENTORY,
    EXPECTED_PRIMARY_ROWS,
    EXPECTED_SEEDS,
    INHERITED_INVENTORY,
    PRIMARY_METHOD,
    REGISTERED_CHOICE_IDS,
    ROOT,
    STUDY_ROOT,
    load_development_inventory,
    materialize_inherited_primary_inventory,
    parent_asset_hashes,
    read_jsonl_gz,
    sha256,
    validate_inventory_shape,
    write_inventory_reports,
    write_jsonl_gz_atomic,
)
from .power import calculate_primary_power
from .primary_scoring import (
    ImageFeatureCache,
    LlavaPrimaryLikelihoodBackend,
    PrimaryScoreCache,
    configure_frozen_tokenizer_runtime,
    prepare_staged_gpu_image_features,
    score_registered_options,
    validate_tokenizer_contract,
)
from .statistics import TASKS, analyze_primary_predictions

STUDY_ID = "PIVOT_EXP_A3P"
DEFAULT_CONFIG = STUDY_ROOT / "config.yaml"
FREEZE_PATH = STUDY_ROOT / "freeze_manifest.yaml"
VALIDATION_ROOT = STUDY_ROOT / "validation"
RESULTS_ROOT = STUDY_ROOT / "results"
OUTPUT_ROOT = ROOT / "outputs/construct_validity/PIVOT_EXP_A3P"
WORK_PREDICTIONS = OUTPUT_ROOT / "predictions.jsonl"
SCORE_CACHE_ROOT = OUTPUT_ROOT / "score_cache"

_PARENT_PRIMARY_SCORE_ANCHORS = {
    "PIVOT_EXP_A3:20260830:sw_20260830_000004:CV2:M0:oracle:neutral_image:forced_choice:shape": {
        "1": -10.173910140991211,
        "2": -11.857503890991211,
        "3": -12.482503890991211,
        "4": -11.709066390991211,
    },
    "PIVOT_EXP_A3:20260830:sw_20260830_000004:CV2:M1:oracle:neutral_image:forced_choice:color": {
        "1": -11.334854125976562,
        "2": -12.553604125976562,
        "3": -14.037979125976562,
        "4": -13.862197875976562,
    },
    (
        "PIVOT_EXP_A3:20260830:sw_20260830_000004:CV2:M2:oracle:"
        "neutral_image:forced_choice:object_identity"
    ): {
        "1": -11.304520606994629,
        "2": -12.409989356994629,
        "3": -12.554520606994629,
        "4": -12.019364356994629,
    },
}

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:64,garbage_collection_threshold:0.70",
)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("PIVOT_EXP_A3P config must be a mapping")
    _validate_config(payload)
    return payload


def prepare_primary_construct_validity(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    run_smoke: bool = True,
) -> dict[str, Any]:
    """Complete every pre-inference gate without reading held-out predictions."""

    config = load_config(config_path)
    create_a3r_final_manifest()
    parents = verify_parent_freezes()
    inheritance = materialize_inherited_primary_inventory()
    write_inventory_reports(inheritance)
    inventory = read_jsonl_gz(INHERITED_INVENTORY)
    inventory_shape = validate_inventory_shape(inventory)
    development = load_development_inventory()
    development_compatibility = _development_compatibility(development, inventory)
    power = calculate_primary_power()
    write_yaml_atomic(STUDY_ROOT / "power_analysis.yaml", power)
    runtime = _runtime_readiness(config)
    write_yaml_atomic(VALIDATION_ROOT / "runtime_readiness.yaml", runtime)
    design_passed = bool(
        parents["passed"]
        and inheritance["passed"]
        and inventory_shape["passed"]
        and development_compatibility["passed"]
        and power["primary_power_passed"]
        and scorer_boundary_report()["passed"]
    )
    if run_smoke and design_passed and runtime["runtime_ready"]:
        smoke = _run_development_primary_smoke(config, development)
    else:
        smoke = {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "status": "NOT_RUN",
            "passed": False,
            "reason": "design_preflight_failed" if not design_passed else "runtime_not_ready",
            "completed_trials": 0,
            "development_accuracy_computed": False,
            "validation_inference_started": False,
        }
        write_yaml_atomic(VALIDATION_ROOT / "development_primary_smoke_report.yaml", smoke)
    passed = bool(design_passed and runtime["runtime_ready"] and smoke["passed"])
    preflight = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "FROZEN_PREINFERENCE"
        if passed
        else (
            "INCONCLUSIVE_PREINFERENCE_POWER"
            if not power["primary_power_passed"]
            else "PREINFERENCE_FAILED"
        ),
        "passed": passed,
        "parent_freezes": parents,
        "parent_primary_immutability": inheritance,
        "inventory_shape": inventory_shape,
        "development_compatibility": development_compatibility,
        "power": power,
        "runtime": runtime,
        "development_primary_smoke": smoke,
        "validation_inference_started": False,
        "scientific_metrics_computed": False,
        "formal_adjudication_run": False,
    }
    write_yaml_atomic(VALIDATION_ROOT / "preflight_report.yaml", preflight)
    freeze = _write_freeze(preflight)
    return {"preflight": preflight, "freeze": freeze}


def run_primary_construct_validity(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    model_name: str = "llava_1_5_7b",
) -> dict[str, Any]:
    """Run exactly 27,000 frozen primary rows with append-only trial-key resume."""

    config = load_config(config_path)
    if model_name != config["model"]["registry_name"]:
        raise ValueError("PIVOT_EXP_A3P must use the preregistered LLaVA model")
    parent_audit = verify_parent_freezes()
    freeze_audit = verify_freeze()
    git_audit = git_execution_boundary()
    if not parent_audit["passed"] or not freeze_audit["passed"] or not git_audit["passed"]:
        raise ValueError(
            f"formal execution boundary failed: parents={parent_audit['passed']} "
            f"freeze={freeze_audit['passed']} git={git_audit['passed']}"
        )
    inventory = read_jsonl_gz(INHERITED_INVENTORY)
    if not validate_inventory_shape(inventory)["passed"]:
        raise ValueError("frozen primary inventory failed immediately before inference")
    existing_rows = _read_work_predictions(WORK_PREDICTIONS)
    existing = {str(row["trial_key"]): row for row in existing_rows}
    if len(existing) != len(existing_rows):
        raise ValueError("working prediction log contains duplicate trial keys")
    expected_keys = {str(row["trial_key"]) for row in inventory}
    if not set(existing).issubset(expected_keys):
        raise ValueError("working prediction log contains a non-frozen trial key")

    tokenizer_runtime = configure_frozen_tokenizer_runtime()
    registry = ModelRegistry(ROOT)
    definition = registry.definition(model_name)
    registered_provenance = _registered_model_provenance(config, definition)
    model = registry.get_or_create(
        definition.experiment_model_config(), seed=int(config["study"]["validation_seeds"][0])
    )
    model.ensure_loaded()
    backend = model._require_backend()  # noqa: SLF001
    model_load = _model_load_report(model, backend)
    tokenizer = validate_tokenizer_contract(backend.tokenizer)
    if not tokenizer["passed"] or not _model_load_passed(model_load):
        raise RuntimeError("formal primary runtime drifted from CUDA/NF4 scoring contract")
    environment = stable_environment_payload(registered_provenance)
    environment_sha = payload_sha256(environment)
    smoke = yaml.safe_load(
        (VALIDATION_ROOT / "development_primary_smoke_report.yaml").read_text(encoding="utf-8")
    )
    if smoke["runtime"]["environment_sha256"] != environment_sha:
        raise ValueError("environment drift since development primary smoke")
    freeze_sha = freeze_audit["freeze_sha256"]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_yaml_atomic(
        OUTPUT_ROOT / "execution_start.yaml",
        {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "status": "validation_inference_started",
            "started_from_commit": git_audit["head"],
            "frozen_preinference_sha256": freeze_sha,
            "environment_sha256": environment_sha,
            "trial_inventory_sha256": sha256(INHERITED_INVENTORY),
            "existing_prediction_count": len(existing_rows),
            "validation_inference_started": True,
            "protocol_change_allowed": False,
        },
    )
    started = time.perf_counter()
    image_hashes = parent_asset_hashes()
    used_images = {
        str(Path(str(row["image"])).resolve()): image_hashes[str(Path(str(row["image"])).resolve())]
        for row in inventory
    }
    images_by_sha = _invert_image_hashes(used_images)
    model_sha = payload_sha256(registered_provenance)
    image_feature_cache = ImageFeatureCache(
        OUTPUT_ROOT / "validation_image_features", model_sha256=model_sha
    )
    staged_runtime = prepare_staged_gpu_image_features(
        model,
        images_by_sha256=images_by_sha,
        feature_cache=image_feature_cache,
        progress_label="PIVOT_EXP_A3P validation",
    )
    score_cache = PrimaryScoreCache(SCORE_CACHE_ROOT)
    scorer = LlavaPrimaryLikelihoodBackend(
        model,
        feature_cache=image_feature_cache,
        image_sha256_by_path=used_images,
    )
    added = 0
    cache_hits = 0
    for trial in inventory:
        trial_key = str(trial["trial_key"])
        if trial_key in existing:
            continue
        prepared = _prepared_from_trial(trial)
        normalized_image = str(Path(str(trial["image"])).resolve())
        image_sha = image_hashes.get(normalized_image)
        if image_sha is None:
            raise ValueError(f"unregistered image at scoring boundary: {trial['image']}")
        cache_key = score_cache.key(
            prompt_sha256=str(trial["prompt_sha256"]),
            image_sha256=image_sha,
            model_sha256=model_sha,
        )
        scored = score_cache.get(cache_key)
        from_cache = scored is not None
        if scored is None:
            scored = score_registered_options(prepared, REGISTERED_CHOICE_IDS, backend=scorer)
            score_cache.put(cache_key, scored)
        else:
            cache_hits += 1
        selected = scored.selected_option_id
        row = {
            **trial,
            **scored.to_dict(),
            "correctness": selected == str(trial["correct_choice_id"]),
            "scored_from_cache": from_cache,
            "freeze_sha256": freeze_sha,
            "environment_sha256": environment_sha,
            "model_provenance": registered_provenance,
            "model_parameter_update": False,
            "image_sha256": image_sha,
        }
        _atomic_append_jsonl(WORK_PREDICTIONS, row)
        existing[trial_key] = row
        added += 1
        if added % 50 == 0:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
            print(
                f"PIVOT_EXP_A3P new={added} total={len(existing)}/{EXPECTED_PRIMARY_ROWS} "
                f"elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )
        seed = int(trial["seed"])
        seed_count = sum(int(value["seed"]) == seed for value in existing.values())
        if seed_count == 5400:
            _write_seed_checkpoint(seed, existing, freeze_sha, environment_sha)

    ordered = [
        existing[str(trial["trial_key"])]
        for trial in inventory
        if str(trial["trial_key"]) in existing
    ]
    integrity = validate_prediction_integrity(
        ordered,
        inventory,
        freeze_sha256=freeze_sha,
        environment_sha256=environment_sha,
    )
    if not integrity["passed"]:
        write_yaml_atomic(OUTPUT_ROOT / "integrity_failure.yaml", integrity)
        raise RuntimeError(f"post-inference primary integrity failed: {integrity}")
    promoted_predictions = RESULTS_ROOT / "predictions.jsonl.gz"
    if promoted_predictions.exists():
        if read_jsonl_gz(promoted_predictions) != ordered:
            raise ValueError("completed promoted predictions differ; overwrite refused")
    else:
        write_jsonl_gz_atomic(promoted_predictions, ordered)
    recovery = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "atomic_append": True,
        "trial_key_resume": True,
        "existing_predictions_at_start": len(existing_rows),
        "new_predictions": added,
        "cache_hits": cache_hits,
        "duplicates": 0,
        "overwrites": 0,
        "per_seed_checkpoints": [seed for seed in EXPECTED_SEEDS],
        "complete": True,
    }
    write_yaml_atomic(RESULTS_ROOT / "execution_recovery.yaml", recovery)
    run = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "INFERENCE_COMPLETE_ADJUDICATION_PENDING",
        "prediction_count": len(ordered),
        "forced_choice_count": len(ordered),
        "complete_validation_seeds": list(EXPECTED_SEEDS),
        "new_predictions": added,
        "elapsed_seconds": time.perf_counter() - started,
        "integrity": integrity,
        "model_provenance": registered_provenance,
        "model_load": model_load,
        "tokenizer_runtime": tokenizer_runtime,
        "staged_runtime": staged_runtime,
        "frozen_preinference_sha256": freeze_sha,
        "environment_sha256": environment_sha,
        "validation_inference_started": True,
        "inference_complete": True,
        "scientific_metrics_computed": False,
        "adjudication_pending": True,
    }
    write_json_atomic(OUTPUT_ROOT / "run.json", run)
    return run


def adjudicate_primary_construct_validity(
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Compute frozen gates only after complete inference and emit the full evidence package."""

    config = load_config(config_path)
    freeze_audit = verify_freeze()
    parent_audit = verify_parent_freezes()
    if not freeze_audit["passed"] or not parent_audit["passed"]:
        raise ValueError("adjudication refused because frozen artifacts drifted")
    predictions_path = RESULTS_ROOT / "predictions.jsonl.gz"
    predictions = read_jsonl_gz(predictions_path)
    inventory = read_jsonl_gz(INHERITED_INVENTORY)
    if len(predictions) != EXPECTED_PRIMARY_ROWS:
        raise ValueError("formal adjudication requires all 27,000 predictions")
    environment_shas = {str(row.get("environment_sha256")) for row in predictions}
    if len(environment_shas) != 1:
        raise ValueError("prediction environment SHA is not unique")
    integrity = validate_prediction_integrity(
        predictions,
        inventory,
        freeze_sha256=freeze_audit["freeze_sha256"],
        environment_sha256=next(iter(environment_shas)),
    )
    analysis = analyze_primary_predictions(predictions, config)
    decision = decide(analysis, integrity=integrity)
    _write_result_assets(analysis, decision, integrity, predictions, config)
    return decision


def _run_development_primary_smoke(
    config: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    report_path = VALIDATION_ROOT / "development_primary_smoke_report.yaml"
    score_path = VALIDATION_ROOT / "development_primary_smoke_scores.jsonl.gz"
    if report_path.is_file() and score_path.is_file():
        existing = yaml.safe_load(report_path.read_text(encoding="utf-8"))
        if existing.get("passed") is True and existing.get("completed_trials") == 432:
            if existing.get("score_evidence_sha256") != sha256(score_path):
                raise ValueError("development smoke evidence hash drift")
            return existing
    tokenizer_runtime = configure_frozen_tokenizer_runtime()
    registry = ModelRegistry(ROOT)
    definition = registry.definition(str(config["model"]["registry_name"]))
    registered_provenance = _registered_model_provenance(config, definition)
    model = registry.get_or_create(
        definition.experiment_model_config(), seed=int(config["study"]["development_seed"])
    )
    started = time.perf_counter()
    model.ensure_loaded()
    backend = model._require_backend()  # noqa: SLF001
    model_load = _model_load_report(model, backend)
    tokenizer = validate_tokenizer_contract(backend.tokenizer)
    model_sha = payload_sha256(registered_provenance)
    development_image_hashes = {
        str(Path(str(row["image"])).resolve()): sha256(row["image"]) for row in rows
    }
    image_feature_cache = ImageFeatureCache(
        OUTPUT_ROOT / "development_image_features", model_sha256=model_sha
    )
    staged_runtime = prepare_staged_gpu_image_features(
        model,
        images_by_sha256=_invert_image_hashes(development_image_hashes),
        feature_cache=image_feature_cache,
        progress_label="PIVOT_EXP_A3P development",
    )
    scorer = LlavaPrimaryLikelihoodBackend(
        model,
        feature_cache=image_feature_cache,
        image_sha256_by_path=development_image_hashes,
    )
    cache = PrimaryScoreCache(OUTPUT_ROOT / "development_score_cache")
    trial_reports = []
    for index, row in enumerate(rows, start=1):
        prepared = _prepared_from_trial(row)
        image_sha = sha256(row["image"])
        first = score_registered_options(prepared, REGISTERED_CHOICE_IDS, backend=scorer)
        second = score_registered_options(prepared, REGISTERED_CHOICE_IDS, backend=scorer)
        key = cache.key(
            prompt_sha256=str(row["prompt_sha256"]),
            image_sha256=image_sha,
            model_sha256=model_sha,
        )
        cache.put(key, first)
        cached = cache.get(key)
        trial_reports.append(
            {
                "trial_key": row["trial_key"],
                "seed": row["seed"],
                "scene_id": row["scene_id"],
                "first": first.to_dict(),
                "second_score_sha256": payload_sha256(second.to_dict()),
                "first_score_sha256": payload_sha256(first.to_dict()),
                "deterministic_repeat_stable": first == second,
                "cache_roundtrip_stable": cached == first,
                "ground_truth_joined": False,
            }
        )
        if index % 12 == 0:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
            print(
                f"PIVOT_EXP_A3P development_primary_smoke={index}/432 "
                f"elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )
    write_jsonl_gz_atomic(score_path, trial_reports)
    four = sum(len(row["first"]["choice_log_likelihoods"]) == 4 for row in trial_reports)
    finite = sum(
        all(math.isfinite(value) for value in row["first"]["choice_log_likelihoods"].values())
        for row in trial_reports
    )
    registered = sum(
        row["first"]["selected_option_id"] in REGISTERED_CHOICE_IDS for row in trial_reports
    )
    deterministic = sum(row["deterministic_repeat_stable"] for row in trial_reports)
    cache_stable = sum(row["cache_roundtrip_stable"] for row in trial_reports)
    observed_by_key = {row["trial_key"]: row["first"] for row in trial_reports}
    anchor_comparison = {
        key: {
            "expected": expected,
            "observed": observed_by_key.get(key, {}).get("choice_log_likelihoods"),
            "exact": observed_by_key.get(key, {}).get("choice_log_likelihoods") == expected,
        }
        for key, expected in _PARENT_PRIMARY_SCORE_ANCHORS.items()
    }
    anchors_exact = all(row["exact"] for row in anchor_comparison.values())
    import torch

    environment = stable_environment_payload(registered_provenance)
    passed = bool(
        len(trial_reports) == four == finite == registered == deterministic == cache_stable == 432
        and tokenizer["passed"]
        and torch.cuda.is_available()
        and _model_load_passed(model_load)
        and staged_runtime["no_cpu_fallback"]
        and anchors_exact
    )
    report = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "PASS" if passed else "PRIMARY_MEASUREMENT_FAILURE",
        "passed": passed,
        "development_seed": 20260830,
        "development_scenes": 8,
        "planned_trials": 432,
        "completed_trials": len(trial_reports),
        "four_score_rate": four / 432,
        "finite_score_rate": finite / 432,
        "registered_prediction_rate": registered / 432,
        "deterministic_repeat_rate": deterministic / 432,
        "cache_roundtrip_rate": cache_stable / 432,
        "parent_primary_score_anchors_exact": anchors_exact,
        "parent_primary_score_anchor_comparison": anchor_comparison,
        "CUDA_available": bool(torch.cuda.is_available()),
        "NF4_loaded": bool(model_load["nf4_loaded"]),
        "CPU_fallback": not bool(model_load["no_cpu_fallback"]),
        "OOM": False,
        "development_accuracy_computed": False,
        "candidate_superiority_computed": False,
        "oracle_corrupted_contrast_computed": False,
        "task_performance_computed": False,
        "validation_inference_started": False,
        "score_evidence_sha256": sha256(score_path),
        "runtime": {
            "model_load": model_load,
            "staged_runtime": staged_runtime,
            "tokenizer_contract": tokenizer,
            "tokenizer_runtime": tokenizer_runtime,
            "environment": environment,
            "environment_sha256": payload_sha256(environment),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_yaml_atomic(report_path, report)
    return report


def _runtime_readiness(config: dict[str, Any]) -> dict[str, Any]:
    import platform

    import bitsandbytes
    import torch
    import transformers

    registry = ModelRegistry(ROOT)
    definition = registry.definition(str(config["model"]["registry_name"]))
    model_config = definition.experiment_model_config()
    model_path = ROOT / str(model_config["model_path"])
    checkpoint_manifest = ROOT / str(model_config["checkpoint_manifest"])
    checks = {
        "cuda_available": bool(torch.cuda.is_available()),
        "checkpoint_directory": model_path.is_dir(),
        "checkpoint_manifest": checkpoint_manifest.is_file(),
        "nf4_registered": model_config.get("quantization") == "nf4",
        "no_cpu_device_map": model_config.get("device_map") == "auto",
        "local_only": model_config.get("local_files_only") is True,
        "revision_frozen": config["model"]["revision"]
        == "4481d270cc22fd5c4d1bb5df129622006ccd9234",
    }
    return {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "runtime_ready": all(checks.values()),
        "checks": checks,
        "python_executable": os.sys.executable,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_built_cuda": str(torch.version.cuda),
        "transformers_version": str(transformers.__version__),
        "bitsandbytes_version": str(bitsandbytes.__version__),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "weights_loaded": False,
        "validation_inference_started": False,
    }


def _write_freeze(preflight: dict[str, Any]) -> dict[str, Any]:
    passed = bool(preflight["passed"])
    locked_paths = (
        "reports/retired_secondary_measurement.md",
        "research/construct_validity/PIVOT_EXP_A3/v1_artifact_manifest.yaml",
        "research/construct_validity/PIVOT_EXP_A3/validation/inventory_manifest.yaml",
        "research/construct_validity/PIVOT_EXP_A3R/final_instrument_retirement_record.yaml",
        "research/construct_validity/PIVOT_EXP_A3R/final_artifact_manifest.yaml",
        "research/construct_validity/PIVOT_EXP_A3P/research_question.md",
        "research/construct_validity/PIVOT_EXP_A3P/instrument_retirement_rationale.md",
        "research/construct_validity/PIVOT_EXP_A3P/hypothesis_registry.yaml",
        "research/construct_validity/PIVOT_EXP_A3P/preregistration.md",
        "research/construct_validity/PIVOT_EXP_A3P/power_analysis.yaml",
        "research/construct_validity/PIVOT_EXP_A3P/config.yaml",
        "research/construct_validity/PIVOT_EXP_A3P/decision_policy.yaml",
        "research/construct_validity/PIVOT_EXP_A3P/protocols/measurement_protocol.md",
        "research/construct_validity/PIVOT_EXP_A3P/protocols/sampling_and_trial_protocol.md",
        "research/construct_validity/PIVOT_EXP_A3P/protocols/statistical_protocol.md",
        "research/construct_validity/PIVOT_EXP_A3P/validation/inherited_primary_inventory.jsonl.gz",
        "research/construct_validity/PIVOT_EXP_A3P/validation/development_primary_inventory.jsonl.gz",
        "research/construct_validity/PIVOT_EXP_A3P/validation/development_inventory_manifest.yaml",
        "research/construct_validity/PIVOT_EXP_A3P/validation/parent_inventory_comparison.yaml",
        "research/construct_validity/PIVOT_EXP_A3P/validation/primary_immutability_report.md",
        "research/construct_validity/PIVOT_EXP_A3P/validation/runtime_readiness.yaml",
        "research/construct_validity/PIVOT_EXP_A3P/validation/development_primary_smoke_report.yaml",
        "research/construct_validity/PIVOT_EXP_A3P/validation/development_primary_smoke_scores.jsonl.gz",
        "research/construct_validity/PIVOT_EXP_A3P/validation/preflight_report.yaml",
        "src/recoalign/cli.py",
        "src/recoalign/construct_validity_primary/__init__.py",
        "src/recoalign/construct_validity_primary/inventory.py",
        "src/recoalign/construct_validity_primary/primary_scoring.py",
        "src/recoalign/construct_validity_primary/power.py",
        "src/recoalign/construct_validity_primary/statistics.py",
        "src/recoalign/construct_validity_primary/integrity.py",
        "src/recoalign/construct_validity_primary/decision.py",
        "src/recoalign/construct_validity_primary/runner.py",
        "src/recoalign/models/vlm/base.py",
        "src/recoalign/models/vlm/llava.py",
        "src/recoalign/models/vlm/registry.py",
        "configs/models/llava_1_5_7b.yaml",
        "manifests/checkpoints/llava_v1_5_7b.yaml",
    )
    locked = {}
    missing = []
    for relative in locked_paths:
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
        else:
            locked[relative] = sha256(path)
    if passed and missing:
        raise FileNotFoundError(f"freeze artifacts missing: {missing}")
    freeze = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "FROZEN_PREINFERENCE" if passed else preflight["status"],
        "parent_primary_immutability": bool(preflight["parent_primary_immutability"]["passed"]),
        "primary_power_passed": bool(preflight["power"]["primary_power_passed"]),
        "development_primary_smoke_passed": bool(preflight["development_primary_smoke"]["passed"]),
        "validation_inventory_frozen": bool(preflight["inventory_shape"]["passed"]),
        "execution_runtime_ready": bool(preflight["runtime"]["runtime_ready"]),
        "validation_inference_started": False,
        "scientific_metrics_computed": False,
        "formal_adjudication_run": False,
        "preinference_commit_required_before_execution": True,
        "locked_artifacts": locked,
        "missing_locked_artifacts": missing,
        "parent_inventory_sha256": sha256(
            ROOT / "research/construct_validity/PIVOT_EXP_A3/validation/trial_inventory.jsonl.gz"
        ),
        "inherited_inventory_sha256": sha256(INHERITED_INVENTORY),
        "development_inventory_sha256": sha256(DEVELOPMENT_INVENTORY),
        "secondary_instrument": {
            "status": "RETIRED_BEFORE_VALIDATION",
            "participates_in_scientific_decision": False,
            "source_study": "PIVOT_EXP_A3R",
        },
    }
    write_yaml_atomic(FREEZE_PATH, freeze)
    return freeze


def _development_compatibility(
    development: list[dict[str, Any]], validation: list[dict[str, Any]]
) -> dict[str, Any]:
    development_seeds = {int(row["seed"]) for row in development}
    validation_seeds = {int(row["seed"]) for row in validation}
    a2_seeds = {20260818, 20260819, 20260820, 20260821, 20260822}
    checks = {
        "development_rows": len(development) == 432,
        "development_validation_disjoint": not (development_seeds & validation_seeds),
        "development_A2_disjoint": not (development_seeds & a2_seeds),
        "validation_A2_disjoint": not (validation_seeds & a2_seeds),
        "exact_measurement_surface": all(
            row["response_method"] == "forced_choice"
            and str(row["prompt"]).endswith("FINAL_CHOICE=")
            and tuple(row["choices"]) == REGISTERED_CHOICE_IDS
            for row in development
        ),
        "all_54_cells_per_scene": Counter(row["scene_id"] for row in development)
        == Counter({scene: 54 for scene in {row["scene_id"] for row in development}}),
    }
    return {
        "checks": checks,
        "development_seed": sorted(development_seeds),
        "validation_seeds": sorted(validation_seeds),
        "A2_seeds": sorted(a2_seeds),
        "passed": all(checks.values()),
    }


def _prepared_from_trial(trial: dict[str, Any]) -> PreparedInput:
    prompt = str(trial["prompt"])
    return PreparedInput(
        prompt=prompt,
        image=str(trial["image"]),
        condition=(f"{trial['manipulation']}_{trial['evidence_truth']}_{trial['image_context']}"),
        setting="construct_validity_primary",
        input_tokens=len(prompt.split()),
        evidence_tokens=len(prompt.split()),
        semantic_units=4,
        relation_count=0,
        semantic_facts_sha256=str(trial["scaffold_sha256"]),
        serialization=str(trial["manipulation"]),
        padding_units=0,
        token_match_delta=0,
        prompt_protocol_id="pivot-exp-a3-answer-contract-v2-primary-inherited",
        prompt_protocol_sha256=None,
    )


def _registered_model_provenance(config: dict[str, Any], definition: Any) -> dict[str, Any]:
    checkpoint_manifest = ROOT / "manifests/checkpoints/llava_v1_5_7b.yaml"
    return {
        "registry_name": config["model"]["registry_name"],
        "model_id": config["model"]["model_id"],
        "revision": config["model"]["revision"],
        "loader": config["model"]["loader"],
        "dtype": config["model"]["dtype"],
        "device_map": config["model"]["device_map"],
        "quantization": config["model"]["quantization"],
        "local_files_only": config["model"]["local_files_only"],
        "model_config_sha256": definition.sha256,
        "checkpoint_manifest_sha256": sha256(checkpoint_manifest),
        "runtime_strategy": ("staged_gpu_vision_feature_cache_then_nf4_language_forward_v1"),
    }


def _invert_image_hashes(image_sha256_by_path: dict[str, str]) -> dict[str, str]:
    images_by_sha: dict[str, str] = {}
    for path, digest in sorted(image_sha256_by_path.items()):
        if digest in images_by_sha and images_by_sha[digest] != path:
            if Path(images_by_sha[digest]).read_bytes() != Path(path).read_bytes():
                raise ValueError("distinct image bytes share a registered SHA identity")
        images_by_sha[digest] = path
    return images_by_sha


def _model_load_report(model: Any, backend: Any) -> dict[str, Any]:
    language_devices = sorted({str(parameter.device) for parameter in backend.model.parameters()})
    vision_devices = sorted(
        {str(parameter.device) for parameter in backend.vision_model.parameters()}
    )
    projector_devices = sorted(
        {str(parameter.device) for parameter in backend.projector.parameters()}
    )
    four_bit_modules = sum(
        type(module).__name__ in {"Linear4bit", "LinearNF4"} for module in backend.model.modules()
    )
    devices = language_devices + vision_devices + projector_devices
    return {
        "weights_loaded": bool(model.loaded and backend.model is not None),
        "vision_tower_loaded": backend.vision_model is not None,
        "projector_loaded": backend.projector is not None,
        "backend": type(backend).__name__,
        "quantization": model.quantization,
        "dtype": model.dtype,
        "device_map": model.device_map,
        "local_files_only": model.local_files_only,
        "is_loaded_in_4bit": bool(getattr(backend.model, "is_loaded_in_4bit", False)),
        "four_bit_module_count": four_bit_modules,
        "language_parameter_devices": language_devices,
        "vision_parameter_devices": vision_devices,
        "projector_parameter_devices": projector_devices,
        "no_cpu_fallback": bool(devices) and all("cpu" not in device for device in devices),
        "nf4_loaded": bool(
            model.quantization == "nf4"
            and getattr(backend.model, "is_loaded_in_4bit", False)
            and four_bit_modules > 0
        ),
    }


def _model_load_passed(report: dict[str, Any]) -> bool:
    return bool(
        report["weights_loaded"]
        and report["vision_tower_loaded"]
        and report["projector_loaded"]
        and report["nf4_loaded"]
        and report["no_cpu_fallback"]
    )


def _atomic_append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")
    binary_flag = getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY | binary_flag, 0o644)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError(f"partial atomic append: {written}/{len(payload)} bytes")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_work_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise ValueError("working prediction log ends in a partial row")
    return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]


def _write_seed_checkpoint(
    seed: int,
    predictions: dict[str, dict[str, Any]],
    freeze_sha: str,
    environment_sha: str,
) -> None:
    rows = [row for row in predictions.values() if int(row["seed"]) == seed]
    write_yaml_atomic(
        OUTPUT_ROOT / "checkpoints" / f"seed_{seed}.yaml",
        {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "seed": seed,
            "completed_rows": len(rows),
            "expected_rows": 5400,
            "complete": len(rows) == 5400,
            "unique_trial_keys": len({row["trial_key"] for row in rows}),
            "freeze_sha256": freeze_sha,
            "environment_sha256": environment_sha,
        },
    )


def _write_result_assets(
    analysis: dict[str, Any],
    decision: dict[str, Any],
    integrity: dict[str, Any],
    predictions: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        RESULTS_ROOT / "metrics.json",
        {"analysis": analysis, "decision": decision, "integrity": integrity},
    )
    write_yaml_atomic(
        RESULTS_ROOT / "primary_measurement_report.yaml",
        {"gate_A": analysis["gate_A"], "integrity": integrity},
    )
    write_yaml_atomic(
        RESULTS_ROOT / "scaffold_comprehension.yaml",
        {name: row["gate_C"] for name, row in analysis["candidates"].items()},
    )
    write_yaml_atomic(
        RESULTS_ROOT / "semantic_sufficiency.yaml",
        {name: row["gate_D"] for name, row in analysis["candidates"].items()},
    )
    write_yaml_atomic(
        RESULTS_ROOT / "intervention_separation.yaml",
        {name: row["gate_E"] for name, row in analysis["candidates"].items()},
    )
    write_yaml_atomic(
        RESULTS_ROOT / "image_interference.yaml",
        {name: row["gate_F"] for name, row in analysis["candidates"].items()},
    )
    write_yaml_atomic(
        RESULTS_ROOT / "task_specific_results.yaml",
        {
            name: {
                "gates": row["gates"],
                "C": row["gate_C"]["by_task"],
                "D": row["gate_D"]["by_task"],
                "E": row["gate_E"]["by_task"],
                "F": row["gate_F"]["by_task"],
            }
            for name, row in analysis["candidates"].items()
        },
    )
    write_yaml_atomic(
        RESULTS_ROOT / "statistical_tests.yaml",
        {
            "bootstrap": config["statistics"],
            "gate_E_holm": {
                name: row["gate_E"]["holm_family"] for name, row in analysis["candidates"].items()
            },
            "gate_F_holm": {
                name: row["gate_F"]["holm_family"] for name, row in analysis["candidates"].items()
            },
            "aggregate_override_allowed": False,
        },
    )
    write_yaml_atomic(
        RESULTS_ROOT / "retired_secondary_reference.yaml",
        {
            "schema_version": 1,
            "source_studies": ["PIVOT_EXP_A3", "PIVOT_EXP_A3R"],
            "status": "RETIRED_BEFORE_VALIDATION",
            "participates_in_scientific_decision": False,
            "development_trials": 160,
            "parsed_trials": 127,
            "unparsed_trials": 33,
            "parse_rate": 0.79375,
            "invalid_outputs": {"E2": 25, "E4": 6, "E1": 2},
            "no_third_repair_attempted": True,
            "open_ended_generation_claim_allowed": False,
        },
    )
    write_yaml_atomic(RESULTS_ROOT / "decision_report.yaml", decision)
    (RESULTS_ROOT / "decision_report.md").write_text(
        _decision_markdown(decision, analysis), encoding="utf-8"
    )
    environment_sha = str(predictions[0]["environment_sha256"])
    write_yaml_atomic(
        RESULTS_ROOT / "provenance.yaml",
        {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "preinference_commit": git_execution_boundary()["head"],
            "freeze_sha256": sha256(FREEZE_PATH),
            "environment_sha256": environment_sha,
            "inventory_sha256": sha256(INHERITED_INVENTORY),
            "predictions_sha256": sha256(RESULTS_ROOT / "predictions.jsonl.gz"),
            "model_provenance": predictions[0]["model_provenance"],
            "prediction_count": len(predictions),
            "validation_seeds": list(EXPECTED_SEEDS),
            "model_parameter_update": False,
        },
    )
    _create_figures(analysis)
    manifest = {}
    for path in sorted(RESULTS_ROOT.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.yaml":
            manifest[path.relative_to(RESULTS_ROOT).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    write_yaml_atomic(
        RESULTS_ROOT / "artifact_manifest.yaml",
        {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "hash_algorithm": "sha256",
            "artifact_count": len(manifest),
            "artifacts": manifest,
            "manifest_self_hash_policy": "excluded_to_avoid_circular_digest",
        },
    )


def _decision_markdown(decision: dict[str, Any], analysis: dict[str, Any]) -> str:
    a4_allowed = str(decision["authorization"]["PIVOT_EXP_A4_preregistration_allowed"]).lower()
    completed_seeds = len(analysis["completed_seeds"])
    return f"""# PIVOT_EXP_A3P Formal Decision

Outcome: **{decision["outcome"]}**

Selected manipulation: **{decision["selected_manipulation"] or "none"}**

{decision["reason"]}

The decision used Gates A, C, D, E, and F only. M0 remained descriptive and no aggregate endpoint
overrode a task-level failure. The retired secondary instrument did not participate.

## Authorization

- PIVOT_EXP_A4 preregistration allowed: {a4_allowed}
- Independent-backbone replication allowed: false
- Model development allowed: false
- Paper writing allowed: false

## Claim boundary

This result concerns answer-option mapping, scaffold comprehension, scene-truth sufficiency,
oracle/corrupted separation, and image interference under the frozen conditional-likelihood
instrument. It does not establish open-ended generation, response-format robustness, a causal
mechanism, or ReCoAlign method effectiveness.

The secondary external-validity instrument failed twice during development (127/160 valid in A3R;
33 entity-ID continuations) and was retired before validation. No third repair was attempted.

Prediction count: {analysis["prediction_count"]:,}. Complete seeds: {completed_seeds}/5.
"""


def _create_figures(analysis: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    root = RESULTS_ROOT / "figures"
    root.mkdir(parents=True, exist_ok=True)
    candidates = ("M1", "M2")
    conditions = ("oracle", "corrupted")
    values = [
        analysis["candidates"][candidate]["gate_A"]["CV1_comprehension"]["by_condition"][condition][
            "mean"
        ]
        for candidate in candidates
        for condition in conditions
    ]
    _bar_figure(root / "gate_A_CV1.png", values, ["M1 O", "M1 C", "M2 O", "M2 C"], "Gate A: CV1")
    for gate, filename, title in (
        ("gate_C", "gate_C_by_task.png", "Gate C by task"),
        ("gate_D", "gate_D_by_task.png", "Gate D by task"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(TASKS))
        for index, candidate in enumerate(candidates):
            means = [
                analysis["candidates"][candidate][gate]["by_task"][task]["mean"] for task in TASKS
            ]
            ax.bar(x + (index - 0.5) * 0.35, means, width=0.35, label=candidate)
        ax.set_xticks(x, TASKS, rotation=20)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / filename, dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(TASKS))
    width = 0.2
    for c_index, candidate in enumerate(candidates):
        separation = analysis["candidates"][candidate]["gate_E"]["by_task"]
        oracle = [
            separation[task]["mean"] + separation[task]["corrupted_condition"]["mean"]
            for task in TASKS
        ]
        corrupted = [separation[task]["corrupted_condition"]["mean"] for task in TASKS]
        ax.bar(x + (c_index * 2 - 1.5) * width, oracle, width, label=f"{candidate} oracle")
        ax.bar(x + (c_index * 2 - 0.5) * width, corrupted, width, label=f"{candidate} corrupted")
    ax.set_xticks(x, TASKS, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_title("Oracle vs corrupted")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(root / "oracle_vs_corrupted.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = []
    labels = []
    for c_index, candidate in enumerate(candidates):
        for t_index, task in enumerate(TASKS):
            value = analysis["candidates"][candidate]["gate_F"]["by_task"][task]
            position = c_index * 5 + t_index
            interval = value["equivalence_interval"]
            ax.errorbar(
                position,
                value["mean"],
                yerr=[[value["mean"] - interval["lower"]], [interval["upper"] - value["mean"]]],
                fmt="o",
            )
            positions.append(position)
            labels.append(f"{candidate} {task}")
    ax.axhspan(-0.05, 0.05, alpha=0.15, color="green")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.set_title("Neutral minus original: equivalence interval")
    fig.tight_layout()
    fig.savefig(root / "neutral_vs_original_equivalence.png", dpi=180)
    plt.close(fig)
    matrix = np.array(
        [
            [
                int(analysis["candidates"][candidate]["gates"][gate])
                for gate in ("A", "C", "D", "E", "F")
            ]
            for candidate in candidates
        ]
    )
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(5), ("A", "C", "D", "E", "F"))
    ax.set_yticks(range(2), candidates)
    for i in range(2):
        for j in range(5):
            ax.text(j, i, "PASS" if matrix[i, j] else "FAIL", ha="center", va="center")
    ax.set_title("M1 vs M2 gate matrix")
    fig.tight_layout()
    fig.savefig(root / "M1_vs_M2_gate_matrix.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for candidate in candidates:
        seed_means = analysis["candidates"][candidate]["gate_D"]["by_task"]["shape"]["seed_means"]
        ax.plot(list(seed_means), list(seed_means.values()), marker="o", label=f"{candidate} shape")
    ax.set_ylim(0, 1)
    ax.set_title("Seed-level replication")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "seed_level_replication.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.plot([0, 1, 2], [0, 0, 0], linewidth=2)
    ax.scatter([0, 1, 2], [0, 0, 0], s=90)
    ax.set_xticks(
        [0, 1, 2], ["A3 v1\ncontract failure", "A3R\n127/160 valid", "Retired\nbefore validation"]
    )
    ax.set_yticks([])
    ax.set_title("Retired secondary instrument timeline")
    fig.tight_layout()
    fig.savefig(root / "retired_secondary_timeline.png", dpi=180)
    plt.close(fig)


def _bar_figure(path: Path, values: list[float], labels: list[str], title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("study", {}).get("id") != STUDY_ID:
        raise ValueError("PIVOT_EXP_A3P study ID drift")
    if config["study"].get("parent_validation_predictions_observed") is not False:
        raise ValueError("parent validation outcomes may not be observed during preregistration")
    if config["study"].get("parent_scientific_metrics_observed") is not False:
        raise ValueError("parent scientific metrics may not be observed during preregistration")
    if config["measurement"]["response_methods"] != [PRIMARY_METHOD]:
        raise ValueError("primary-only response method drift")
    if tuple(config["measurement"]["registered_option_ids"]) != REGISTERED_CHOICE_IDS:
        raise ValueError("registered option IDs drift")
    if config["measurement"]["prompt_completion_prefix"] != "FINAL_CHOICE=":
        raise ValueError("primary completion boundary drift")
    if config["measurement"]["runtime_strategy"] != (
        "staged_gpu_vision_feature_cache_then_nf4_language_forward_v1"
    ):
        raise ValueError("registered primary runtime strategy drift")
    if config["measurement"]["cpu_model_parameter_execution"] is not False:
        raise ValueError("CPU model parameter execution is prohibited")
    if config["secondary_gate"] != {
        "status": "RETIRED_BEFORE_VALIDATION",
        "participates_in_scientific_decision": False,
        "source_study": "PIVOT_EXP_A3R",
        "reason": "retired invalid external-validity instrument",
    }:
        raise ValueError("secondary retirement disposition drift")
    expected_gates = {
        "A_primary_answer_validity": {
            "valid_measurement_rate": 1.0,
            "CV1_per_condition_mean_minimum": 0.95,
            "CV1_per_condition_confidence_interval_lower_minimum": 0.90,
        },
        "C_scaffold_comprehension": {
            "per_task_mean_minimum": 0.95,
            "per_task_confidence_interval_lower_minimum": 0.90,
        },
        "D_scene_semantic_sufficiency": {
            "per_task_mean_minimum": 0.90,
            "per_task_confidence_interval_lower_minimum": 0.85,
        },
        "E_intervention_separation": {
            "per_task_oracle_minus_corrupted_minimum": 0.50,
            "per_task_confidence_interval_lower_strictly_above": 0.0,
            "corrupted_may_not_pass_gate_D": True,
        },
        "F_image_interference": {
            "paired_TOST_margin": 0.05,
            "equivalence_requires_TOST": True,
            "non_significance_is_not_equivalence": True,
        },
    }
    if config["gates"] != expected_gates:
        raise ValueError("registered Gate A/C/D/E/F thresholds drifted")
    if config["design"]["selection_candidates"] != ["M1", "M2"]:
        raise ValueError("candidate set drift")
    if not all(bool(value) for value in config["prohibitions"].values()):
        raise ValueError("all primary-only prohibitions must remain enabled")


__all__ = [
    "adjudicate_primary_construct_validity",
    "load_config",
    "prepare_primary_construct_validity",
    "run_primary_construct_validity",
]
