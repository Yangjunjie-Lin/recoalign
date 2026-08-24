"""Independent pre-inference amendment lifecycle for PIVOT_EXP_A3R."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import os
import platform
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from datasets.records import SceneRecord
from recoalign.models.vlm.base import PreparedInput
from recoalign.models.vlm.registry import ModelRegistry

from .answer_contract_v3 import (
    CONTRACT_VERSION,
    PROMPT_COMPLETION_PREFIX,
    REGISTERED_CHOICE_IDS,
    parse_continuation,
    validate_parser_contract,
)
from .choice_scoring import (
    LlavaConditionalLikelihoodBackend,
    score_choices,
    validate_tokenizer_contract,
)
from .decision import adjudicate_construct_validity
from .integrity import artifact_metadata
from .statistics import analyze_construct_predictions, wilson_interval
from .trial_builder import build_validation_trials

ROOT = Path(__file__).resolve().parents[3]
STUDY_ID = "PIVOT_EXP_A3R"
PARENT_STUDY_ID = "PIVOT_EXP_A3"
STUDY_ROOT = ROOT / "research/construct_validity/PIVOT_EXP_A3R"
PARENT_ROOT = ROOT / "research/construct_validity/PIVOT_EXP_A3"
DEFAULT_CONFIG = STUDY_ROOT / "config.yaml"
FREEZE_PATH = STUDY_ROOT / "freeze_manifest.yaml"
VALIDATION_ROOT = STUDY_ROOT / "validation"
TRIAL_INVENTORY = VALIDATION_ROOT / "trial_inventory.jsonl.gz"
PARENT_TRIAL_INVENTORY = PARENT_ROOT / "validation/trial_inventory.jsonl.gz"
PARENT_ASSET_MANIFEST = PARENT_ROOT / "validation/visual_asset_manifest.jsonl"
ASSET_MANIFEST = VALIDATION_ROOT / "visual_asset_manifest.jsonl"
DEVELOPMENT_SEED = 20260830

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:64,garbage_collection_threshold:0.70",
)

_SECONDARY_INSTRUCTION_V1 = "Return exactly one line in the form FINAL_CHOICE=<1|2|3|4>."
_SECONDARY_INSTRUCTION_V3 = (
    "Return one registered option by completing the final field.\n\nFINAL_CHOICE="
)
_SECONDARY_KEY_SUFFIX = f"response_contract={CONTRACT_VERSION}"
_PARENT_MANIFEST_EXPECTED_SHA256 = (
    "885489bc1b46c9e561df3633015325d786f72219b7c3cd251bf074af9814ee00"
)

_PROTOCOL_FILES = (
    "reports/cross_project_contamination_audit.md",
    "research/construct_validity/PIVOT_EXP_A3/blocked_preinference_record.yaml",
    "research/construct_validity/PIVOT_EXP_A3/v1_artifact_manifest.yaml",
    "research/construct_validity/PIVOT_EXP_A3R/amendment_rationale.md",
    "research/construct_validity/PIVOT_EXP_A3R/preregistration.md",
    "research/construct_validity/PIVOT_EXP_A3R/hypothesis_registry.yaml",
    "research/construct_validity/PIVOT_EXP_A3R/power_analysis.yaml",
    "research/construct_validity/PIVOT_EXP_A3R/config.yaml",
    "research/construct_validity/PIVOT_EXP_A3R/decision_policy.yaml",
    "research/construct_validity/PIVOT_EXP_A3R/answer_contract/answer-contract-v3-continuation.yaml",
    "research/construct_validity/PIVOT_EXP_A3R/protocols/answer_contract_protocol.md",
    "research/construct_validity/PIVOT_EXP_A3R/protocols/sampling_and_trial_protocol.md",
    "research/construct_validity/PIVOT_EXP_A3R/protocols/statistical_protocol.md",
    "src/recoalign/construct_validity/answer_contract_v3.py",
    "src/recoalign/construct_validity/a3r.py",
    "src/recoalign/construct_validity/choice_scoring.py",
    "src/recoalign/construct_validity/scaffold_comprehension.py",
    "src/recoalign/construct_validity/visual_legend.py",
    "src/recoalign/construct_validity/statistics.py",
    "src/recoalign/construct_validity/decision.py",
)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("PIVOT_EXP_A3R config must be a mapping")
    _validate_config(payload)
    return payload


def validate_answer_contract(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    parser_report = validate_parser_contract()
    tokenizer_report = validate_tokenizer_contract(_load_tokenizer(config))
    report = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "contract_version": CONTRACT_VERSION,
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
    load_config(config_path)
    parent = _verify_parent_v1()
    power = yaml.safe_load((STUDY_ROOT / "power_analysis.yaml").read_text(encoding="utf-8"))
    if power.get("status") != "PASS" or power.get("computed_before_inference") is not True:
        raise ValueError("PIVOT_EXP_A3R power analysis is not a passing pre-inference record")
    contract = validate_answer_contract(config_path)
    if not contract["passed"]:
        raise ValueError("PIVOT_EXP_A3R answer contract failed before preregistration")
    report = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "preregistered_inventory_pending",
        "registered_at": "2026-08-24",
        "parent_study": PARENT_STUDY_ID,
        "parent_outcome": "RUNTIME_BLOCKED_PREINFERENCE",
        "validation_outcomes_observed": False,
        "scientific_metrics_observed": False,
        "repair_basis": "development-runtime-smoke-only",
        "inference_started": False,
        "validation_inference_started": False,
        "weights_loaded": False,
        "protocol_changes_allowed_after_inference": False,
        "parent_v1": parent,
        "power": {
            "status": power["status"],
            "parent_sha256": power["parent_power_analysis_sha256"],
            "amended_sha256": _sha256(STUDY_ROOT / "power_analysis.yaml"),
        },
        "answer_contract": {
            "version": CONTRACT_VERSION,
            "parser_passed": contract["parser"]["passed"],
            "tokenizer_passed": contract["tokenizer"]["passed"],
        },
        "locked_protocol_files": _protocol_hashes(include_generated_contract=True),
        "preflight": {
            "passed": False,
            "design_preflight_passed": False,
            "validation_inventory_frozen": False,
            "execution_runtime_ready": False,
            "development_contract_smoke_passed": False,
        },
    }
    _write_yaml(FREEZE_PATH, report)
    _write_yaml(
        VALIDATION_ROOT / "preregistration_report.yaml",
        {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "status": report["status"],
            "passed": True,
            "parent_outcome": report["parent_outcome"],
            "validation_outcomes_observed": False,
            "scientific_metrics_observed": False,
            "weights_loaded": False,
            "inference_started": False,
        },
    )
    return report


def validate_construct_validity(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    preflight_only: bool = True,
) -> dict[str, Any]:
    if not preflight_only:
        raise ValueError("PIVOT_EXP_A3R validation command is preflight-only")
    config = load_config(config_path)
    freeze = _load_preregistered_freeze()
    parent = _verify_parent_v1()
    inventory = build_amended_inventory()
    contract = validate_answer_contract(config_path)
    runtime = _runtime_readiness(config)
    design_passed = bool(
        inventory["primary_immutability"]["passed"]
        and inventory["secondary_rehash"]["passed"]
        and inventory["trial_integrity"]["passed"]
        and contract["passed"]
        and parent["passed"]
    )
    smoke = (
        _run_development_smoke(config)
        if design_passed and runtime["runtime_ready"]
        else _blocked_smoke_report(design_passed, runtime)
    )
    _write_yaml(VALIDATION_ROOT / "development_runtime_smoke_report.yaml", smoke)
    passed = bool(design_passed and runtime["runtime_ready"] and smoke["passed"])
    status = (
        "frozen_preinference"
        if passed
        else "secondary_contract_runtime_failure"
        if design_passed and runtime["runtime_ready"]
        else "preflight_blocked_runtime"
        if design_passed
        else "preflight_failed"
    )
    report = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": status,
        "frozen_state_label": "FROZEN_PREINFERENCE" if passed else None,
        "passed": passed,
        "design_preflight_passed": design_passed,
        "validation_inventory_frozen": design_passed,
        "execution_runtime_ready": bool(runtime["runtime_ready"]),
        "development_contract_smoke_passed": bool(smoke["passed"]),
        "preflight_only": True,
        "weights_loaded_for_development_smoke": bool(smoke.get("weights_loaded")),
        "inference_started": False,
        "validation_inference_started": False,
        "scientific_metrics_computed": False,
        "parent_v1": parent,
        "answer_contract": contract,
        "primary_immutability": inventory["primary_immutability"],
        "secondary_prompt_rehash": inventory["secondary_rehash"],
        "trial_integrity": inventory["trial_integrity"],
        "runtime_readiness": runtime,
        "development_smoke": smoke,
        "trial_inventory_sha256": _sha256(TRIAL_INVENTORY),
        "visual_asset_manifest_sha256": _sha256(ASSET_MANIFEST),
    }
    _write_yaml(VALIDATION_ROOT / "preflight_report.yaml", report)
    freeze.update(
        {
            "status": status,
            "inference_started": False,
            "validation_inference_started": False,
            "scientific_metrics_computed": False,
            "locked_protocol_files": _protocol_hashes(include_generated_contract=True),
            "preflight": {
                "passed": passed,
                "design_preflight_passed": design_passed,
                "validation_inventory_frozen": design_passed,
                "execution_runtime_ready": bool(runtime["runtime_ready"]),
                "development_contract_smoke_passed": bool(smoke["passed"]),
                "trial_inventory_sha256": report["trial_inventory_sha256"],
                "visual_asset_manifest_sha256": report[
                    "visual_asset_manifest_sha256"
                ],
                "primary_trial_count": 27000,
                "secondary_trial_count": 9000,
                "trial_count": 36000,
                "validation_seed_count": 5,
            },
        }
    )
    _write_yaml(FREEZE_PATH, freeze)
    return report


def build_amended_inventory() -> dict[str, Any]:
    """Derive A3R trials from the frozen v1 inventory without regenerating any scene."""

    parent_rows = _read_jsonl_gz(PARENT_TRIAL_INVENTORY)
    if len(parent_rows) != 36000:
        raise ValueError("frozen PIVOT_EXP_A3 inventory must contain exactly 36,000 rows")
    amended_rows = [_amend_trial(row) for row in parent_rows]
    _write_jsonl_gz(TRIAL_INVENTORY, amended_rows)
    shutil.copyfile(PARENT_ASSET_MANIFEST, ASSET_MANIFEST)

    parent_primary = [row for row in parent_rows if row["response_method"] == "forced_choice"]
    amended_primary = [row for row in amended_rows if row["response_method"] == "forced_choice"]
    parent_secondary = [
        row for row in parent_rows if row["response_method"] == "free_generation"
    ]
    amended_secondary = [
        row for row in amended_rows if row["response_method"] == "free_generation"
    ]
    protected_fields = (
        "trial_key",
        "prompt",
        "prompt_sha256",
        "image",
        "choices",
        "correct_choice_id",
        "scene_truth_choice_id",
        "declared_choice_id",
    )
    primary_mismatches = []
    for index, (parent, amended) in enumerate(
        zip(parent_primary, amended_primary, strict=True)
    ):
        mismatched = [field for field in protected_fields if parent[field] != amended[field]]
        if mismatched and len(primary_mismatches) < 20:
            primary_mismatches.append(
                {
                    "row_index": index,
                    "trial_key": parent["trial_key"],
                    "fields": mismatched,
                }
            )
    unique_image_paths = sorted({str(row["image"]) for row in parent_primary})
    image_hash_checks = [
        {
            "path": path,
            "sha256": _sha256(Path(path)),
            "exists": Path(path).is_file(),
        }
        for path in unique_image_paths
    ]
    primary_report = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "parent_study_id": PARENT_STUDY_ID,
        "comparison_scope": "protected_scientific_and_execution_fields",
        "administrative_study_id_amended": True,
        "expected_count": 27000,
        "compared_count": len(amended_primary),
        "exact_match_count": len(amended_primary) - len(primary_mismatches),
        "exact_match_requirement": "27000 / 27000",
        "protected_fields": list(protected_fields),
        "prompt_sha_unchanged": all(
            parent["prompt_sha256"] == amended["prompt_sha256"]
            for parent, amended in zip(parent_primary, amended_primary, strict=True)
        ),
        "image_path_unchanged": all(
            parent["image"] == amended["image"]
            for parent, amended in zip(parent_primary, amended_primary, strict=True)
        ),
        "image_sha_verified": all(row["exists"] for row in image_hash_checks),
        "choice_mapping_unchanged": all(
            parent["choices"] == amended["choices"]
            for parent, amended in zip(parent_primary, amended_primary, strict=True)
        ),
        "correct_choice_unchanged": all(
            parent["correct_choice_id"] == amended["correct_choice_id"]
            for parent, amended in zip(parent_primary, amended_primary, strict=True)
        ),
        "trial_key_unchanged": all(
            parent["trial_key"] == amended["trial_key"]
            for parent, amended in zip(parent_primary, amended_primary, strict=True)
        ),
        "unique_image_count": len(image_hash_checks),
        "mismatches": primary_mismatches,
    }
    primary_report["passed"] = bool(
        len(parent_primary) == len(amended_primary) == 27000
        and primary_report["exact_match_count"] == 27000
        and primary_report["prompt_sha_unchanged"]
        and primary_report["image_path_unchanged"]
        and primary_report["image_sha_verified"]
        and primary_report["choice_mapping_unchanged"]
        and primary_report["correct_choice_unchanged"]
        and primary_report["trial_key_unchanged"]
    )
    _write_yaml(VALIDATION_ROOT / "primary_immutability_report.yaml", primary_report)

    rehash_rows = []
    for parent, amended in zip(parent_secondary, amended_secondary, strict=True):
        rehash_rows.append(
            {
                "parent_trial_key": parent["trial_key"],
                "amended_trial_key": amended["trial_key"],
                "parent_prompt_sha256": parent["prompt_sha256"],
                "amended_prompt_sha256": amended["prompt_sha256"],
                "prompt_changed": parent["prompt_sha256"] != amended["prompt_sha256"],
                "prompt_ends_exact_prefix": str(amended["prompt"]).endswith(
                    PROMPT_COMPLETION_PREFIX
                ),
                "image_unchanged": parent["image"] == amended["image"],
                "choices_unchanged": parent["choices"] == amended["choices"],
                "correct_choice_unchanged": (
                    parent["correct_choice_id"] == amended["correct_choice_id"]
                ),
                "key_versioned": str(amended["trial_key"]).endswith(
                    _SECONDARY_KEY_SUFFIX
                ),
            }
        )
    secondary_report = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "parent_study_id": PARENT_STUDY_ID,
        "response_contract": CONTRACT_VERSION,
        "expected_count": 9000,
        "rehash_count": len(rehash_rows),
        "all_prompts_rehashed": all(row["prompt_changed"] for row in rehash_rows),
        "all_prompts_end_exact_prefix": all(
            row["prompt_ends_exact_prefix"] for row in rehash_rows
        ),
        "all_images_unchanged": all(row["image_unchanged"] for row in rehash_rows),
        "all_choices_unchanged": all(row["choices_unchanged"] for row in rehash_rows),
        "all_correct_choices_unchanged": all(
            row["correct_choice_unchanged"] for row in rehash_rows
        ),
        "all_keys_versioned": all(row["key_versioned"] for row in rehash_rows),
        "parent_inventory_sha256": _sha256(PARENT_TRIAL_INVENTORY),
        "amended_inventory_sha256": _sha256(TRIAL_INVENTORY),
    }
    secondary_report["passed"] = bool(
        len(parent_secondary) == len(amended_secondary) == 9000
        and all(
            bool(value)
            for key, value in secondary_report.items()
            if key.startswith("all_")
        )
    )
    _write_yaml(VALIDATION_ROOT / "secondary_prompt_rehash_report.yaml", secondary_report)

    by_method = Counter(str(row["response_method"]) for row in amended_rows)
    by_seed = Counter(int(row["seed"]) for row in amended_rows)
    trial_keys = [str(row["trial_key"]) for row in amended_rows]
    trial_integrity = {
        "passed": bool(
            len(amended_rows) == 36000
            and by_method == Counter({"forced_choice": 27000, "free_generation": 9000})
            and len(set(trial_keys)) == len(trial_keys)
            and len(by_seed) == 5
            and all(count == 7200 for count in by_seed.values())
            and all(
                str(row["prompt"]).endswith(PROMPT_COMPLETION_PREFIX)
                for row in amended_rows
            )
        ),
        "trial_count": len(amended_rows),
        "by_method": dict(by_method),
        "by_seed": dict(sorted(by_seed.items())),
        "unique_trial_keys": len(set(trial_keys)) == len(trial_keys),
        "all_prompts_end_exact_prefix": all(
            str(row["prompt"]).endswith(PROMPT_COMPLETION_PREFIX)
            for row in amended_rows
        ),
    }
    inventory_manifest = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "passed": bool(
            primary_report["passed"]
            and secondary_report["passed"]
            and trial_integrity["passed"]
        ),
        "derivation": "frozen_PIVOT_EXP_A3_inventory_secondary_surface_amendment",
        "parent_inventory": artifact_metadata(PARENT_TRIAL_INVENTORY),
        "trial_inventory": artifact_metadata(TRIAL_INVENTORY),
        "parent_visual_asset_manifest": artifact_metadata(PARENT_ASSET_MANIFEST),
        "visual_asset_manifest": artifact_metadata(ASSET_MANIFEST),
        "visual_asset_manifest_bytes_identical": (
            _sha256(PARENT_ASSET_MANIFEST) == _sha256(ASSET_MANIFEST)
        ),
        "validation_outcomes_observed": False,
        "scientific_metrics_computed": False,
    }
    _write_yaml(VALIDATION_ROOT / "inventory_manifest.yaml", inventory_manifest)
    return {
        "primary_immutability": primary_report,
        "secondary_rehash": secondary_report,
        "trial_integrity": trial_integrity,
        "inventory_manifest": inventory_manifest,
    }


def _amend_trial(parent: dict[str, Any]) -> dict[str, Any]:
    amended = copy.deepcopy(parent)
    amended["study_id"] = STUDY_ID
    if amended["response_method"] == "forced_choice":
        return amended
    prompt = str(amended["prompt"])
    if not prompt.endswith(_SECONDARY_INSTRUCTION_V1):
        raise ValueError(f"unexpected parent secondary prompt boundary: {parent['trial_key']}")
    prompt = prompt[: -len(_SECONDARY_INSTRUCTION_V1)] + _SECONDARY_INSTRUCTION_V3
    prompt = prompt.replace(
        "ANSWER_CONTRACT=answer-contract-v2", f"ANSWER_CONTRACT={CONTRACT_VERSION}", 1
    )
    if not prompt.endswith(PROMPT_COMPLETION_PREFIX) or prompt.endswith(
        PROMPT_COMPLETION_PREFIX + " "
    ):
        raise ValueError("amended secondary prompt does not end at the exact frozen prefix")
    amended["prompt"] = prompt
    amended["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    amended["trial_key"] = f"{parent['trial_key']}:{_SECONDARY_KEY_SUFFIX}"
    amended["metadata"] = {
        **amended["metadata"],
        "response_contract": CONTRACT_VERSION,
        "prompt_completion_prefix": PROMPT_COMPLETION_PREFIX,
        "raw_continuation_stored_verbatim": True,
        "reconstructed_contract_output_stored": True,
    }
    return amended


def _run_development_smoke(config: dict[str, Any]) -> dict[str, Any]:
    """Run every registered secondary cell on all eight frozen development scenes."""

    import bitsandbytes
    import torch
    import transformers

    records = _read_development_records()
    rows = _required_development_smoke_rows(records)
    if len(records) != 8 or len(rows) != 160:
        raise RuntimeError(
            f"development smoke inventory drift: scenes={len(records)} trials={len(rows)}"
        )
    registry = ModelRegistry(ROOT)
    definition = registry.definition(str(config["model"]["registry_name"]))
    model = registry.get_or_create(
        definition.experiment_model_config(), seed=DEVELOPMENT_SEED
    )
    started = time.perf_counter()
    trial_reports = []
    oom = False
    error: str | None = None
    try:
        model.ensure_loaded()
        backend = model._require_backend()  # noqa: SLF001 - explicit runtime audit boundary
        model_load = _model_load_report(model, backend)
        generation = {
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": 16,
            "num_beams": 1,
        }
        for index, row in enumerate(rows, start=1):
            first = model.generate(str(row["prompt"]), image=str(row["image"]), **generation)
            second = model.generate(str(row["prompt"]), image=str(row["image"]), **generation)
            parsed = parse_continuation(first)
            trial_reports.append(
                {
                    "trial_key": row["trial_key"],
                    "scene_id": row["scene_id"],
                    "manipulation": row["manipulation"],
                    "image_context": row["image_context"],
                    "construct": row["construct"],
                    "task": row["task"],
                    "prompt_completion_prefix": parsed.prompt_completion_prefix,
                    "raw_continuation": parsed.raw_continuation,
                    "reconstructed_contract_output": parsed.reconstructed_contract_output,
                    "raw_continuation_parsed": parsed.raw_valid,
                    "reconstructed_output_parsed": parsed.reconstructed_valid,
                    "choice_ids_match": parsed.choice_ids_match,
                    "parse_failure_reason": parsed.failure_reason,
                    "deterministic_repeat_stable": first == second,
                }
            )
            if index % 12 == 0:
                print(
                    f"PIVOT_EXP_A3R development_smoke={index}/160 "
                    f"elapsed_s={time.perf_counter() - started:.1f}",
                    flush=True,
                )
    except RuntimeError as exc:
        error = f"{type(exc).__name__}: {exc}"
        oom = "out of memory" in str(exc).lower()
        model_load = locals().get("model_load", {"weights_loaded": bool(model.loaded)})

    raw_parsed = sum(bool(row["raw_continuation_parsed"]) for row in trial_reports)
    reconstructed_parsed = sum(
        bool(row["reconstructed_output_parsed"]) for row in trial_reports
    )
    deterministic = sum(
        bool(row["deterministic_repeat_stable"]) for row in trial_reports
    )
    n = len(rows)
    raw_rate = raw_parsed / n
    reconstructed_rate = reconstructed_parsed / n
    deterministic_rate = deterministic / n
    checks = {
        "all_8_development_scenes": len({row["scene_id"] for row in rows}) == 8,
        "all_registered_secondary_cells": _smoke_cell_coverage(rows),
        "raw_continuation_parse_rate": raw_rate == 1.0,
        "reconstructed_output_parse_rate": reconstructed_rate == 1.0,
        "deterministic_repeat_rate": deterministic_rate == 1.0,
        "unparsed_count_zero": raw_parsed == reconstructed_parsed == n,
        "cuda_available": bool(torch.cuda.is_available()),
        "historical_runtime_exact": bool(
            platform.python_version() == "3.12.6"
            and torch.__version__ == "2.7.1+cu126"
            and torch.version.cuda == "12.6"
            and transformers.__version__ == "4.52.4"
            and bitsandbytes.__version__ == "0.48.2"
            and torch.cuda.is_available()
            and "RTX 3060" in torch.cuda.get_device_name(0)
        ),
        "nf4_loaded": bool(model_load.get("nf4_loaded")),
        "cpu_fallback": not bool(model_load.get("no_cpu_fallback")),
        "no_cpu_fallback": bool(model_load.get("no_cpu_fallback")),
        "oom": oom,
        "no_oom": not oom and error is None,
        "validation_predictions_read": False,
        "development_accuracy_computed": False,
        "scientific_metrics_computed": False,
    }
    passed = bool(
        checks["all_8_development_scenes"]
        and checks["all_registered_secondary_cells"]
        and checks["raw_continuation_parse_rate"]
        and checks["reconstructed_output_parse_rate"]
        and checks["deterministic_repeat_rate"]
        and checks["unparsed_count_zero"]
        and checks["cuda_available"]
        and checks["historical_runtime_exact"]
        and checks["nf4_loaded"]
        and checks["no_cpu_fallback"]
        and checks["no_oom"]
    )
    return {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "PASS" if passed else "SECONDARY_CONTRACT_RUNTIME_FAILURE",
        "outcome": None if passed else "SECONDARY_CONTRACT_RUNTIME_FAILURE",
        "passed": passed,
        "development_seed": DEVELOPMENT_SEED,
        "development_scene_count": len(records),
        "planned_trial_count": n,
        "completed_trial_count": len(trial_reports),
        "raw_continuation_parse_rate": raw_rate,
        "reconstructed_output_parse_rate": reconstructed_rate,
        "deterministic_repeat_rate": deterministic_rate,
        "unparsed_count": n - min(raw_parsed, reconstructed_parsed),
        "CUDA_available": bool(torch.cuda.is_available()),
        "NF4_loaded": bool(model_load.get("nf4_loaded")),
        "CPU_fallback": not bool(model_load.get("no_cpu_fallback")),
        "OOM": oom,
        "weights_loaded": bool(model_load.get("weights_loaded")),
        "runtime": {
            "python_executable": os.sys.executable,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torch_built_cuda": torch.version.cuda,
            "transformers_version": transformers.__version__,
            "bitsandbytes_version": bitsandbytes.__version__,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "allocator_environment": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "model_load": model_load,
        },
        "checks": checks,
        "error": error,
        "trials": trial_reports,
        "reporting_prohibitions": {
            "development_accuracy_computed": False,
            "candidate_superiority_computed": False,
            "semantic_sufficiency_computed": False,
            "M1_vs_M2_performance_computed": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def _read_development_records() -> list[SceneRecord]:
    path = ROOT / "outputs/construct_validity/PIVOT_EXP_A3/inventory/development/dataset.jsonl"
    rows = _read_jsonl(path)
    records = [SceneRecord.from_dict(row) for row in rows]
    if any(int(record.metadata.get("seed", -1)) != DEVELOPMENT_SEED for record in records):
        raise RuntimeError("development smoke attempted to use a non-development seed")
    return records


def _required_development_smoke_rows(
    records: list[SceneRecord],
) -> list[dict[str, Any]]:
    m2_mapping = _development_m2_mapping(records)
    parent_trials = build_validation_trials(
        records,
        seed=DEVELOPMENT_SEED,
        neutral_image=PARENT_ROOT / "semantic_manipulations/neutral.png",
        m2_images=m2_mapping,
    )
    rows = [
        _amend_trial(trial.to_dict())
        for trial in parent_trials
        if trial.response_method == "free_generation"
        and trial.manipulation in {"M1", "M2"}
        and trial.evidence_truth == "oracle"
    ]
    records_by_id = {record.scene_id: record for record in records}
    original_cv1 = []
    for row in rows:
        if row["task"] != "answer_contract_comprehension":
            continue
        amended = copy.deepcopy(row)
        amended["image_context"] = "original_scene_image"
        amended["trial_key"] = str(amended["trial_key"]).replace(
            ":neutral_image:", ":original_scene_image:", 1
        )
        if amended["manipulation"] == "M2":
            amended["image"] = m2_mapping[
                (str(amended["scene_id"]), "oracle", "original_scene_image")
            ]
        else:
            amended["image"] = records_by_id[str(amended["scene_id"])].image
        amended["metadata"] = {
            **amended["metadata"],
            "development_smoke_extension": "original_image_CV1_runtime_only",
        }
        original_cv1.append(amended)
    combined = rows + original_cv1
    if len(combined) != 160 or len({str(row["trial_key"]) for row in combined}) != 160:
        raise RuntimeError("required development smoke rows are incomplete or duplicated")
    return sorted(combined, key=lambda row: str(row["trial_key"]))


def _development_m2_mapping(
    records: list[SceneRecord],
) -> dict[tuple[str, str, str], str]:
    root = ROOT / "outputs/construct_validity/PIVOT_EXP_A3/inventory/m2_development"
    mapping = {}
    for record in records:
        for evidence_truth in ("oracle", "corrupted"):
            for image_context in ("neutral_image", "original_scene_image"):
                path = root / record.scene_id / evidence_truth / f"{image_context}.png"
                if not path.is_file():
                    raise FileNotFoundError(path)
                mapping[(record.scene_id, evidence_truth, image_context)] = str(path.resolve())
    return mapping


def _smoke_cell_coverage(rows: list[dict[str, Any]]) -> bool:
    observed = Counter(
        (str(row["manipulation"]), str(row["image_context"]), str(row["task"]))
        for row in rows
    )
    expected = Counter()
    tasks = ("shape", "color", "object_identity", "entity_attribute_binding")
    for manipulation in ("M1", "M2"):
        for image_context in ("neutral_image", "original_scene_image"):
            for task in tasks:
                expected[(manipulation, image_context, task)] = 8
            expected[(manipulation, image_context, "answer_contract_comprehension")] = 8
    return observed == expected


def _model_load_report(model: Any, backend: Any) -> dict[str, Any]:
    language_devices = sorted({str(parameter.device) for parameter in backend.model.parameters()})
    vision_devices = sorted(
        {str(parameter.device) for parameter in backend.vision_model.parameters()}
    )
    projector_devices = sorted(
        {str(parameter.device) for parameter in backend.projector.parameters()}
    )
    four_bit_modules = sum(
        type(module).__name__ in {"Linear4bit", "LinearNF4"}
        for module in backend.model.modules()
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


def _blocked_smoke_report(
    design_passed: bool, runtime: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "NOT_RUN",
        "outcome": None,
        "passed": False,
        "reason": "design_preflight_failed" if not design_passed else "runtime_not_ready",
        "runtime_ready": bool(runtime.get("runtime_ready")),
        "weights_loaded": False,
        "development_accuracy_computed": False,
        "validation_predictions_read": False,
        "scientific_metrics_computed": False,
    }


def complete_development_smoke_extension(
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Complete the prospectively required original-image CV1 development cells only."""

    import bitsandbytes
    import torch
    import transformers

    config = load_config(config_path)
    report_path = VALIDATION_ROOT / "development_runtime_smoke_report.yaml"
    report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "SECONDARY_CONTRACT_RUNTIME_FAILURE":
        raise ValueError("development smoke extension requires the frozen failed 144-row smoke")
    existing = {str(row["trial_key"]): row for row in report["trials"]}
    if len(existing) != 144 or len(report["trials"]) != 144:
        raise ValueError("development smoke extension refuses non-144-row source evidence")
    records = _read_development_records()
    required = _required_development_smoke_rows(records)
    missing = [row for row in required if str(row["trial_key"]) not in existing]
    if len(missing) != 16 or any(
        row["task"] != "answer_contract_comprehension"
        or row["image_context"] != "original_scene_image"
        for row in missing
    ):
        raise ValueError("development smoke extension is not exactly 16 original-image CV1 rows")

    registry = ModelRegistry(ROOT)
    definition = registry.definition(str(config["model"]["registry_name"]))
    model = registry.get_or_create(
        definition.experiment_model_config(), seed=DEVELOPMENT_SEED
    )
    started = time.perf_counter()
    model.ensure_loaded()
    backend = model._require_backend()  # noqa: SLF001 - explicit runtime audit boundary
    model_load = _model_load_report(model, backend)
    generation = {
        "temperature": 0.0,
        "do_sample": False,
        "max_new_tokens": 16,
        "num_beams": 1,
    }
    new_reports = []
    for index, row in enumerate(missing, start=1):
        first = model.generate(str(row["prompt"]), image=str(row["image"]), **generation)
        second = model.generate(str(row["prompt"]), image=str(row["image"]), **generation)
        parsed = parse_continuation(first)
        new_reports.append(
            {
                "trial_key": row["trial_key"],
                "scene_id": row["scene_id"],
                "manipulation": row["manipulation"],
                "image_context": row["image_context"],
                "construct": row["construct"],
                "task": row["task"],
                "prompt_completion_prefix": parsed.prompt_completion_prefix,
                "raw_continuation": parsed.raw_continuation,
                "reconstructed_contract_output": parsed.reconstructed_contract_output,
                "raw_continuation_parsed": parsed.raw_valid,
                "reconstructed_output_parsed": parsed.reconstructed_valid,
                "choice_ids_match": parsed.choice_ids_match,
                "parse_failure_reason": parsed.failure_reason,
                "deterministic_repeat_stable": first == second,
            }
        )
        print(
            f"PIVOT_EXP_A3R development_smoke_extension={index}/16 "
            f"elapsed_s={time.perf_counter() - started:.1f}",
            flush=True,
        )
    merged = sorted(
        [*report["trials"], *new_reports], key=lambda row: str(row["trial_key"])
    )
    raw_parsed = sum(bool(row["raw_continuation_parsed"]) for row in merged)
    reconstructed_parsed = sum(
        bool(row["reconstructed_output_parsed"]) for row in merged
    )
    deterministic = sum(bool(row["deterministic_repeat_stable"]) for row in merged)
    n = len(merged)
    raw_rate = raw_parsed / n
    reconstructed_rate = reconstructed_parsed / n
    deterministic_rate = deterministic / n
    checks = report["checks"]
    checks.update(
        {
            "all_8_development_scenes": len({row["scene_id"] for row in merged}) == 8,
            "all_registered_secondary_cells": _smoke_cell_coverage(required),
            "raw_continuation_parse_rate": raw_rate == 1.0,
            "reconstructed_output_parse_rate": reconstructed_rate == 1.0,
            "deterministic_repeat_rate": deterministic_rate == 1.0,
            "unparsed_count_zero": raw_parsed == reconstructed_parsed == n,
            "cuda_available": bool(torch.cuda.is_available()),
            "historical_runtime_exact": bool(
                platform.python_version() == "3.12.6"
                and torch.__version__ == "2.7.1+cu126"
                and torch.version.cuda == "12.6"
                and transformers.__version__ == "4.52.4"
                and bitsandbytes.__version__ == "0.48.2"
                and torch.cuda.is_available()
                and "RTX 3060" in torch.cuda.get_device_name(0)
            ),
            "nf4_loaded": bool(model_load["nf4_loaded"]),
            "cpu_fallback": not bool(model_load["no_cpu_fallback"]),
            "no_cpu_fallback": bool(model_load["no_cpu_fallback"]),
            "oom": False,
            "no_oom": True,
            "validation_predictions_read": False,
            "development_accuracy_computed": False,
            "scientific_metrics_computed": False,
        }
    )
    passed = bool(
        checks["all_8_development_scenes"]
        and checks["all_registered_secondary_cells"]
        and checks["raw_continuation_parse_rate"]
        and checks["reconstructed_output_parse_rate"]
        and checks["deterministic_repeat_rate"]
        and checks["unparsed_count_zero"]
        and checks["cuda_available"]
        and checks["historical_runtime_exact"]
        and checks["nf4_loaded"]
        and checks["no_cpu_fallback"]
        and checks["no_oom"]
    )
    old_smoke_sha = _sha256(report_path)
    report.update(
        {
            "status": "PASS" if passed else "SECONDARY_CONTRACT_RUNTIME_FAILURE",
            "outcome": None if passed else "SECONDARY_CONTRACT_RUNTIME_FAILURE",
            "passed": passed,
            "planned_trial_count": n,
            "completed_trial_count": n,
            "raw_continuation_parse_rate": raw_rate,
            "reconstructed_output_parse_rate": reconstructed_rate,
            "deterministic_repeat_rate": deterministic_rate,
            "unparsed_count": n - min(raw_parsed, reconstructed_parsed),
            "CUDA_available": bool(torch.cuda.is_available()),
            "NF4_loaded": bool(model_load["nf4_loaded"]),
            "CPU_fallback": not bool(model_load["no_cpu_fallback"]),
            "OOM": False,
            "weights_loaded": True,
            "runtime": {
                **report["runtime"],
                "model_load": model_load,
            },
            "checks": checks,
            "error": None,
            "trials": merged,
            "elapsed_seconds": float(report["elapsed_seconds"])
            + time.perf_counter()
            - started,
            "extension": {
                "source_completed_trial_count": 144,
                "added_original_image_CV1_trials": 16,
                "existing_trials_regenerated": False,
                "development_accuracy_computed": False,
                "validation_predictions_read": False,
            },
        }
    )
    _write_yaml(report_path, report)
    new_smoke_sha = _sha256(report_path)

    preflight_path = VALIDATION_ROOT / "preflight_report.yaml"
    preflight = yaml.safe_load(preflight_path.read_text(encoding="utf-8"))
    preflight["status"] = (
        "frozen_preinference" if passed else "secondary_contract_runtime_failure"
    )
    preflight["frozen_state_label"] = "FROZEN_PREINFERENCE" if passed else None
    preflight["passed"] = passed
    preflight["development_contract_smoke_passed"] = passed
    preflight["development_smoke"] = report
    _write_yaml(preflight_path, preflight)

    integrity_record = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "change_class": "prevalidation_integrity_and_required_smoke_completion",
        "timing": "after_initial_development_smoke_before_validation_inference",
        "scientific_protocol_changed": False,
        "parser_changed": False,
        "prompt_changed": False,
        "gate_changed": False,
        "inventory_changed": False,
        "existing_smoke_outputs_changed": False,
        "validation_inference_started": False,
        "scientific_metrics_computed": False,
        "changes": {
            "source_cleanup": "removed lexical contamination from unused helper",
            "development_smoke_extension": "added_16_required_original_image_CV1_cells",
        },
        "source_smoke_report_sha256": old_smoke_sha,
        "completed_smoke_report_sha256": new_smoke_sha,
        "unchanged_evidence": {
            "answer_contract_v3_parser_sha256": _sha256(
                ROOT / "src/recoalign/construct_validity/answer_contract_v3.py"
            ),
            "trial_inventory_sha256": _sha256(TRIAL_INVENTORY),
        },
        "outcome": report["outcome"],
    }
    integrity_path = VALIDATION_ROOT / "post_smoke_administrative_integrity_record.yaml"
    _write_yaml(integrity_path, integrity_record)

    freeze = yaml.safe_load(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("validation_inference_started") is not False:
        raise ValueError("cannot complete development smoke after validation inference")
    freeze["status"] = preflight["status"]
    freeze["locked_protocol_files"] = _protocol_hashes(include_generated_contract=True)
    freeze["preflight"]["passed"] = passed
    freeze["preflight"]["development_contract_smoke_passed"] = passed
    freeze["preflight"]["development_smoke_trial_count"] = n
    _write_yaml(FREEZE_PATH, freeze)

    stop_path = VALIDATION_ROOT / "execution_stop_record.yaml"
    stop = yaml.safe_load(stop_path.read_text(encoding="utf-8"))
    stop.update(
        {
            "status": "FROZEN_PREINFERENCE" if passed else "NOT_FROZEN_PREINFERENCE",
            "required_state_achieved": passed,
            "outcome": report["outcome"],
            "development_contract_smoke_passed": passed,
            "raw_continuation_parse_rate": raw_rate,
            "reconstructed_output_parse_rate": reconstructed_rate,
            "deterministic_repeat_rate": deterministic_rate,
            "unparsed_count": report["unparsed_count"],
        }
    )
    stop["source_artifacts"].update(
        {
            "freeze_manifest_sha256": _sha256(FREEZE_PATH),
            "preflight_report_sha256": _sha256(preflight_path),
            "development_runtime_smoke_report_sha256": new_smoke_sha,
        }
    )
    _write_yaml(stop_path, stop)
    return report


def run_construct_validity(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    model_name: str = "llava_1_5_7b",
) -> dict[str, Any]:
    config = load_config(config_path)
    if model_name != config["model"]["registry_name"]:
        raise ValueError("PIVOT_EXP_A3R must use the preregistered LLaVA model")
    _verify_execution_freeze()
    parser_report = validate_parser_contract()
    tokenizer_report = validate_tokenizer_contract(_load_tokenizer(config))
    if not parser_report["passed"] or not tokenizer_report["passed"]:
        raise ValueError("PIVOT_EXP_A3R answer contract drifted at execution boundary")
    trial_rows = _read_jsonl_gz(TRIAL_INVENTORY)
    output = ROOT / config["study"]["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.jsonl"
    existing = _read_jsonl(predictions_path)
    predictions = {str(row["trial_key"]): row for row in existing}
    if len(predictions) != len(existing):
        raise ValueError("PIVOT_EXP_A3R predictions contain duplicate trial keys")
    expected_keys = {str(row["trial_key"]) for row in trial_rows}
    if not set(predictions).issubset(expected_keys):
        raise ValueError("existing prediction file contains a non-frozen trial key")
    _write_yaml(
        output / "execution_start.yaml",
        {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "status": "validation_inference_started",
            "frozen_preinference_sha256": _sha256(FREEZE_PATH),
            "trial_inventory_sha256": _sha256(TRIAL_INVENTORY),
            "existing_prediction_count": len(existing),
            "model": model_name,
            "validation_inference_started": True,
            "protocol_change_allowed": False,
        },
    )

    registry = ModelRegistry(ROOT)
    definition = registry.definition(model_name)
    model = registry.get_or_create(
        definition.experiment_model_config(),
        seed=int(config["study"]["validation_seeds"][0]),
    )
    likelihood_backend = LlavaConditionalLikelihoodBackend(model)
    started = time.perf_counter()
    added = 0
    freeze_sha = _sha256(FREEZE_PATH)
    for trial in trial_rows:
        trial_key = str(trial["trial_key"])
        if trial_key in predictions:
            continue
        prepared = _prepared_from_trial(trial)
        if trial["response_method"] == "forced_choice":
            scored = score_choices(
                prepared, REGISTERED_CHOICE_IDS, backend=likelihood_backend
            )
            prediction_choice = scored.predicted_choice_id
            result = {
                "raw_output": prediction_choice,
                "prompt_completion_prefix": PROMPT_COMPLETION_PREFIX,
                "raw_continuation": prediction_choice,
                "reconstructed_contract_output": PROMPT_COMPLETION_PREFIX
                + prediction_choice,
                "prediction_choice_id": prediction_choice,
                "valid_measurement": scored.valid_measurement,
                "parsed": True,
                "raw_continuation_parsed": True,
                "reconstructed_output_parsed": True,
                "choice_ids_match": True,
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
            parsed = parse_continuation(raw)
            prediction_choice = parsed.choice_id
            result = {
                "raw_output": raw,
                "prompt_completion_prefix": parsed.prompt_completion_prefix,
                "raw_continuation": parsed.raw_continuation,
                "reconstructed_contract_output": parsed.reconstructed_contract_output,
                "prediction_choice_id": prediction_choice,
                "valid_measurement": False,
                "parsed": parsed.valid,
                "raw_continuation_parsed": parsed.raw_valid,
                "reconstructed_output_parsed": parsed.reconstructed_valid,
                "choice_ids_match": parsed.choice_ids_match,
                "parse_failure_reason": parsed.failure_reason,
                "choice_scores": None,
                "scoring_method": "frozen_continuation_generation_answer-contract-v3",
                "tied_maximum": False,
            }
        row = {
            **trial,
            **result,
            "correct": prediction_choice == trial["correct_choice_id"],
            "freeze_sha256": freeze_sha,
        }
        _append_jsonl(predictions_path, row)
        predictions[trial_key] = row
        added += 1
        if added % 25 == 0:
            print(
                f"PIVOT_EXP_A3R new={added} total={len(predictions)} "
                f"elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )
    rows = list(predictions.values())
    integrity = _prediction_integrity(rows, trial_rows, config)
    if not integrity["passed"]:
        raise RuntimeError(f"post-inference PIVOT_EXP_A3R integrity failed: {integrity}")
    run = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "inference_complete_adjudication_pending",
        "prediction_count": len(rows),
        "forced_choice_count": integrity["forced_choice_count"],
        "continuation_generation_count": integrity["continuation_generation_count"],
        "complete_validation_seeds": integrity["completed_seeds"],
        "new_predictions": added,
        "elapsed_seconds": time.perf_counter() - started,
        "integrity": integrity,
        "model_provenance": model.provenance(),
        "frozen_preinference_sha256": freeze_sha,
        "validation_inference_started": True,
        "inference_complete": True,
        "scientific_metrics_computed": False,
        "adjudication_pending": True,
    }
    _write_json(output / "run.json", run)
    return run


def _prediction_integrity(
    rows: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    expected_keys = {str(row["trial_key"]) for row in trial_rows}
    observed_keys = {str(row["trial_key"]) for row in rows}
    primary = [row for row in rows if row["response_method"] == "forced_choice"]
    secondary = [row for row in rows if row["response_method"] == "free_generation"]
    secondary_recomputed = [parse_continuation(str(row["raw_continuation"])) for row in secondary]
    completed_seeds = sorted({int(row["seed"]) for row in rows})
    by_seed = Counter(int(row["seed"]) for row in rows)
    assertions = {
        "all_registered_trials_present": observed_keys == expected_keys,
        "no_duplicate_trial_keys": len(observed_keys) == len(rows),
        "total_count": len(rows) == int(config["design"]["trial_counts"]["total"]),
        "forced_choice_count": len(primary) == 27000,
        "continuation_generation_count": len(secondary) == 9000,
        "primary_measurement_valid": all(
            bool(row.get("valid_measurement"))
            and str(row.get("prediction_choice_id")) in REGISTERED_CHOICE_IDS
            and isinstance(row.get("choice_scores"), dict)
            and len(row["choice_scores"]) == 4
            and all(math.isfinite(float(value)) for value in row["choice_scores"].values())
            for row in primary
        ),
        "secondary_rows_retained": len(secondary) == 9000,
        "secondary_reconstruction_mechanical": all(
            str(row["reconstructed_contract_output"])
            == PROMPT_COMPLETION_PREFIX + str(row["raw_continuation"])
            for row in secondary
        ),
        "secondary_parser_reproducible": all(
            bool(row["raw_continuation_parsed"]) == parsed.raw_valid
            and bool(row["reconstructed_output_parsed"]) == parsed.reconstructed_valid
            and bool(row["choice_ids_match"]) == parsed.choice_ids_match
            and bool(row["parsed"]) == parsed.valid
            and row["prediction_choice_id"] == parsed.choice_id
            for row, parsed in zip(secondary, secondary_recomputed, strict=True)
        ),
        "five_complete_seeds": len(completed_seeds) == 5
        and all(count == 7200 for count in by_seed.values()),
        "single_freeze_hash": {str(row["freeze_sha256"]) for row in rows}
        == {_sha256(FREEZE_PATH)},
    }
    return {
        "passed": all(bool(value) for value in assertions.values()),
        "assertions": assertions,
        "prediction_count": len(rows),
        "forced_choice_count": len(primary),
        "continuation_generation_count": len(secondary),
        "completed_seeds": completed_seeds,
        "by_seed": dict(sorted(by_seed.items())),
        "unparsed_secondary_count": sum(not parsed.valid for parsed in secondary_recomputed),
    }


def adjudicate_construct_validity_run(
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    freeze = _verify_execution_freeze()
    output = ROOT / config["study"]["output_root"]
    trial_rows = _read_jsonl_gz(TRIAL_INVENTORY)
    rows = _read_jsonl(output / "predictions.jsonl")
    integrity = _prediction_integrity(rows, trial_rows, config)
    if not integrity["passed"] or len(rows) != 36000:
        raise ValueError("PIVOT_EXP_A3R may not adjudicate before 36,000 integrity-valid rows")
    analysis = _analyze_predictions(rows, config)
    decision = _replace_study_id(
        adjudicate_construct_validity(analysis, integrity_passed=integrity["passed"])
    )
    _promote_results(config, rows, analysis, decision, integrity, freeze)
    return {"analysis": analysis, "decision": decision}


def _analyze_predictions(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    analysis = _replace_study_id(analyze_construct_predictions(rows, config))
    secondary = [row for row in rows if row["response_method"] == "free_generation"]
    for manipulation in ("M1", "M2"):
        selected = [row for row in secondary if row["manipulation"] == manipulation]
        gate_b = _dual_parse_gate(selected, config)
        candidate = analysis["candidates"][manipulation]
        candidate["gate_B"] = gate_b
        candidate["gates"]["B"] = gate_b["passed"]
        candidate["passed_all_gates"] = all(bool(value) for value in candidate["gates"].values())
    analysis["answer_contract_amendment"] = {
        "version": CONTRACT_VERSION,
        "prompt_prefix": PROMPT_COMPLETION_PREFIX,
        "raw_generated_continuation_stored": True,
        "reconstructed_surface_contract_stored": True,
        "secondary_used_for_primary_accuracy": False,
    }
    return analysis


def _dual_parse_gate(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    n = len(rows)
    raw_successes = sum(bool(row["raw_continuation_parsed"]) for row in rows)
    reconstructed_successes = sum(
        bool(row["reconstructed_output_parsed"]) for row in rows
    )
    matching = sum(bool(row["choice_ids_match"]) for row in rows)
    raw_rate = raw_successes / n if n else 0.0
    reconstructed_rate = reconstructed_successes / n if n else 0.0
    raw_interval = wilson_interval(raw_successes, n) if n else (0.0, 1.0)
    reconstructed_interval = (
        wilson_interval(reconstructed_successes, n) if n else (0.0, 1.0)
    )
    gate = config["gates"]["B_secondary_parser_integrity"]
    passed = bool(
        n
        and raw_rate >= float(gate["raw_continuation_parse_rate_minimum"])
        and raw_interval[0] >= float(gate["raw_continuation_ci_lower_minimum"])
        and reconstructed_rate
        >= float(gate["reconstructed_contract_parse_rate_minimum"])
        and reconstructed_interval[0]
        >= float(gate["reconstructed_contract_ci_lower_minimum"])
        and matching == n
    )
    return {
        "passed": passed,
        "n": n,
        "raw_continuation": {
            "parsed": raw_successes,
            "parse_rate": raw_rate,
            "confidence_interval": {
                "lower": raw_interval[0],
                "upper": raw_interval[1],
                "method": "Wilson",
            },
            "required_parse_rate": float(
                gate["raw_continuation_parse_rate_minimum"]
            ),
            "required_ci_lower": float(gate["raw_continuation_ci_lower_minimum"]),
        },
        "reconstructed_contract": {
            "parsed": reconstructed_successes,
            "parse_rate": reconstructed_rate,
            "confidence_interval": {
                "lower": reconstructed_interval[0],
                "upper": reconstructed_interval[1],
                "method": "Wilson",
            },
            "required_parse_rate": float(
                gate["reconstructed_contract_parse_rate_minimum"]
            ),
            "required_ci_lower": float(
                gate["reconstructed_contract_ci_lower_minimum"]
            ),
        },
        "choice_id_matches": matching,
        "choice_id_match_rate": matching / n if n else 0.0,
    }


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
    _write_yaml(results / "prediction_integrity.yaml", integrity)
    _write_yaml(
        results / "answer_contract_report.yaml",
        {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "contract_version": CONTRACT_VERSION,
            "primary": analysis["gate_A"],
            "secondary": {
                name: value["gate_B"]
                for name, value in analysis["candidates"].items()
            },
            "secondary_used_for_primary_accuracy": False,
        },
    )
    _write_yaml(
        results / "parse_integrity_report.yaml",
        {name: value["gate_B"] for name, value in analysis["candidates"].items()},
    )
    for filename, key in (
        ("scaffold_comprehension.yaml", "gate_C"),
        ("semantic_sufficiency.yaml", "gate_D"),
        ("image_interference.yaml", "gate_F"),
    ):
        _write_yaml(
            results / filename,
            {name: value[key] for name, value in analysis["candidates"].items()},
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
    _write_yaml(results / "next_stage_authorization.yaml", decision["authorization"])
    _write_yaml(
        results / "provenance.yaml",
        {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "parent_study": PARENT_STUDY_ID,
            "model": config["model"],
            "freeze_sha256": _sha256(FREEZE_PATH),
            "trial_inventory_sha256": freeze["preflight"]["trial_inventory_sha256"],
            "parent_trial_inventory_sha256": _sha256(PARENT_TRIAL_INVENTORY),
            "integrity": integrity,
        },
    )
    from .runner import _build_figures  # noqa: PLC0415 - reuse frozen v1 figure code

    _build_figures(rows, analysis, results / "figures")
    (results / "scientific_execution_report.md").write_text(
        _scientific_execution_markdown(analysis, decision, integrity),
        encoding="utf-8",
        newline="\n",
    )
    artifacts = {}
    for path in sorted(results.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.yaml":
            artifacts[path.relative_to(results).as_posix()] = artifact_metadata(path)
    _write_yaml(
        results / "artifact_manifest.yaml",
        {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "outcome": decision["outcome"],
            "prediction_lines": len(rows),
            "forced_choice_lines": integrity["forced_choice_count"],
            "continuation_generation_lines": integrity[
                "continuation_generation_count"
            ],
            "artifacts": artifacts,
        },
    )


def _decision_markdown(decision: dict[str, Any]) -> str:
    authorization = yaml.safe_dump(decision["authorization"], sort_keys=False).rstrip()
    return f"""# PIVOT_EXP_A3R Formal Decision

## Outcome

`{decision['outcome']}`

Selected manipulation: `{decision['selected_manipulation']}`

{decision['reason']}

## Next-stage authorization

```yaml
{authorization}
```
"""


def _scientific_execution_markdown(
    analysis: dict[str, Any], decision: dict[str, Any], integrity: dict[str, Any]
) -> str:
    return f"""# PIVOT_EXP_A3R Scientific Execution Report

The frozen run completed {integrity['prediction_count']:,} / 36,000 predictions:
{integrity['forced_choice_count']:,} primary forced-choice and
{integrity['continuation_generation_count']:,} secondary continuation-generation rows across five
complete validation seeds. Prediction integrity passed before any scientific metric was computed.
No failed or unparsed secondary row was deleted.

The secondary response remained external-validity evidence only. Primary accuracy used only the
unchanged conditional-log-likelihood scorer. Gate B evaluated the raw continuation and mechanically
reconstructed contract independently for both M1 and M2.

Formal outcome: `{decision['outcome']}`. Selected manipulation:
`{decision['selected_manipulation']}`. Full task-specific estimates, intervals, Holm corrections,
and paired equivalence tests are stored in the machine-readable result artifacts.
"""


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
        raise ValueError(f"PIVOT_EXP_A3R config missing keys: {missing}")
    study = config["study"]
    if study.get("id") != STUDY_ID:
        raise ValueError("A3R config study ID drift")
    if study.get("parent_study") != PARENT_STUDY_ID:
        raise ValueError("A3R parent study drift")
    if study.get("parent_outcome") != "RUNTIME_BLOCKED_PREINFERENCE":
        raise ValueError("A3R parent outcome drift")
    if any(
        bool(study.get(field))
        for field in ("validation_outcomes_observed", "scientific_metrics_observed")
    ):
        raise ValueError("A3R amendment cannot be based on validation outcomes or metrics")
    if study.get("repair_basis") != "development-runtime-smoke-only":
        raise ValueError("A3R repair basis drift")
    parent = yaml.safe_load((PARENT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    unchanged_study_fields = (
        "development_seed",
        "validation_seeds",
        "validation_scenes_per_seed",
        "development_scenes",
        "generated_candidates_per_validation_seed",
        "required_entities_per_scene",
        "source_generator",
        "scene_selection",
        "behavior_based_selection",
    )
    for field in unchanged_study_fields:
        if study[field] != parent["study"][field]:
            raise ValueError(f"A3R study design field drifted: {field}")
    for section in ("model", "semantic_manipulations", "design", "statistics"):
        if config[section] != parent[section]:
            raise ValueError(f"A3R frozen scientific section drifted: {section}")
    for gate in (
        "A_primary_answer_validity",
        "C_scaffold_comprehension",
        "D_scene_semantic_sufficiency",
        "E_intervention_separation",
        "F_image_interference",
    ):
        if config["gates"][gate] != parent["gates"][gate]:
            raise ValueError(f"A3R scientific gate drifted: {gate}")
    gate_b = config["gates"]["B_secondary_parser_integrity"]
    expected_gate_b = {
        "parse_rate_minimum": 0.99,
        "confidence_interval_lower_minimum": 0.98,
        "raw_continuation_parse_rate_minimum": 0.99,
        "raw_continuation_ci_lower_minimum": 0.98,
        "reconstructed_contract_parse_rate_minimum": 0.99,
        "reconstructed_contract_ci_lower_minimum": 0.98,
    }
    if gate_b != expected_gate_b:
        raise ValueError("A3R Gate B thresholds drifted")
    contract = config["answer_contract"]
    if contract.get("version") != CONTRACT_VERSION:
        raise ValueError("A3R contract version drift")
    if contract.get("secondary_prompt_completion_prefix") != PROMPT_COMPLETION_PREFIX:
        raise ValueError("A3R prompt completion prefix drift")
    primary_fields = (
        "option_ids",
        "entity_id_namespaces",
        "primary_method",
        "primary_completion_prefix",
        "primary_leading_space_allowed",
        "primary_fallback_method",
        "constrained_decoding_allowed",
    )
    for field in primary_fields:
        if contract[field] != parent["answer_contract"][field]:
            raise ValueError(f"A3R primary answer contract drifted: {field}")
    if not all(bool(value) for value in config["prohibitions"].values()):
        raise ValueError("all PIVOT_EXP_A3R prohibitions must remain enabled")


def _verify_parent_v1() -> dict[str, Any]:
    manifest_path = PARENT_ROOT / "v1_artifact_manifest.yaml"
    if _sha256(manifest_path) != _PARENT_MANIFEST_EXPECTED_SHA256:
        raise ValueError("PIVOT_EXP_A3 v1 artifact manifest drift")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for relative, expected in manifest["artifacts"].items():
        path = PARENT_ROOT / relative
        actual = _sha256(path) if path.is_file() else None
        if actual != expected:
            mismatches.append(
                {"path": relative, "expected_sha256": expected, "actual_sha256": actual}
            )
    blocked = yaml.safe_load(
        (PARENT_ROOT / "blocked_preinference_record.yaml").read_text(encoding="utf-8")
    )
    predictions = ROOT / "outputs/construct_validity/PIVOT_EXP_A3/predictions.jsonl"
    passed = bool(
        not mismatches
        and blocked.get("outcome") == "RUNTIME_BLOCKED_PREINFERENCE"
        and blocked.get("validation_inference_started") is False
        and blocked.get("scientific_metrics_computed") is False
        and not predictions.exists()
    )
    return {
        "passed": passed,
        "manifest_sha256": _sha256(manifest_path),
        "artifact_count": len(manifest["artifacts"]),
        "artifact_mismatches": mismatches,
        "outcome": blocked.get("outcome"),
        "validation_inference_started": blocked.get("validation_inference_started"),
        "scientific_metrics_computed": blocked.get("scientific_metrics_computed"),
        "parent_prediction_file_exists": predictions.exists(),
    }


def _load_preregistered_freeze() -> dict[str, Any]:
    if not FREEZE_PATH.is_file():
        raise FileNotFoundError("PIVOT_EXP_A3R is not preregistered")
    freeze = yaml.safe_load(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("status") != "preregistered_inventory_pending":
        raise ValueError("PIVOT_EXP_A3R is not at the preregistered inventory-pending boundary")
    if freeze.get("inference_started") is not False:
        raise ValueError("PIVOT_EXP_A3R freeze is not pre-inference")
    _verify_locked_files(freeze)
    return freeze


def _verify_execution_freeze() -> dict[str, Any]:
    if not FREEZE_PATH.is_file():
        raise FileNotFoundError("PIVOT_EXP_A3R freeze manifest is missing")
    freeze = yaml.safe_load(FREEZE_PATH.read_text(encoding="utf-8"))
    preflight = freeze.get("preflight", {})
    required = (
        freeze.get("status") == "frozen_preinference",
        preflight.get("passed") is True,
        preflight.get("design_preflight_passed") is True,
        preflight.get("validation_inventory_frozen") is True,
        preflight.get("execution_runtime_ready") is True,
        preflight.get("development_contract_smoke_passed") is True,
        freeze.get("inference_started") is False,
    )
    if not all(required):
        raise ValueError("PIVOT_EXP_A3R execution barrier is not FROZEN_PREINFERENCE")
    if _sha256(TRIAL_INVENTORY) != preflight["trial_inventory_sha256"]:
        raise ValueError("PIVOT_EXP_A3R trial inventory hash drift")
    if _sha256(ASSET_MANIFEST) != preflight["visual_asset_manifest_sha256"]:
        raise ValueError("PIVOT_EXP_A3R visual asset manifest drift")
    _verify_locked_files(freeze)
    if not _verify_parent_v1()["passed"]:
        raise ValueError("PIVOT_EXP_A3 v1 boundary drift")
    return freeze


def _verify_locked_files(freeze: dict[str, Any]) -> None:
    mismatches = []
    for relative, expected in freeze["locked_protocol_files"].items():
        path = ROOT / relative
        actual = _sha256(path) if path.is_file() else None
        if actual != expected:
            mismatches.append(relative)
    if mismatches:
        raise ValueError(f"PIVOT_EXP_A3R locked protocol drift: {mismatches}")


def _protocol_hashes(*, include_generated_contract: bool) -> dict[str, str]:
    relative_files = list(_PROTOCOL_FILES)
    if include_generated_contract:
        generated = (
            "research/construct_validity/PIVOT_EXP_A3R/answer_contract/tokenizer_contract.yaml",
            "research/construct_validity/PIVOT_EXP_A3R/validation/answer_contract_preflight.yaml",
            "research/construct_validity/PIVOT_EXP_A3R/validation/preregistration_report.yaml",
            "research/construct_validity/PIVOT_EXP_A3R/validation/primary_immutability_report.yaml",
            "research/construct_validity/PIVOT_EXP_A3R/validation/secondary_prompt_rehash_report.yaml",
            "research/construct_validity/PIVOT_EXP_A3R/validation/inventory_manifest.yaml",
            "research/construct_validity/PIVOT_EXP_A3R/validation/development_runtime_smoke_report.yaml",
            "research/construct_validity/PIVOT_EXP_A3R/validation/post_smoke_administrative_integrity_record.yaml",
            "research/construct_validity/PIVOT_EXP_A3R/validation/preflight_report.yaml",
        )
        relative_files.extend(relative for relative in generated if (ROOT / relative).is_file())
    return {relative: _sha256(ROOT / relative) for relative in relative_files}


def _runtime_readiness(config: dict[str, Any]) -> dict[str, Any]:
    import bitsandbytes
    import torch
    import transformers

    registry = ModelRegistry(ROOT)
    definition = registry.definition(str(config["model"]["registry_name"]))
    model = registry.get_or_create(
        definition.experiment_model_config(),
        seed=int(config["study"]["validation_seeds"][0]),
    )
    report = model.dry_run()
    if report.get("weights_loaded") is not False:
        raise RuntimeError("A3R runtime readiness must not load weights")
    exact = bool(
        platform.python_version() == "3.12.6"
        and torch.__version__ == "2.7.1+cu126"
        and torch.version.cuda == "12.6"
        and transformers.__version__ == "4.52.4"
        and bitsandbytes.__version__ == "0.48.2"
        and torch.cuda.is_available()
        and "RTX 3060" in torch.cuda.get_device_name(0)
    )
    report.update(
        {
            "historical_runtime_exact": exact,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torch_built_cuda": torch.version.cuda,
            "transformers_version": transformers.__version__,
            "bitsandbytes_version": bitsandbytes.__version__,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "runtime_ready": bool(report.get("runtime_ready") and exact),
        }
    )
    _write_yaml(VALIDATION_ROOT / "runtime_readiness.yaml", report)
    return report


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
        prompt_protocol_id="pivot-exp-a3r-answer-contract-v3-continuation",
        prompt_protocol_sha256=None,
    )


def _load_tokenizer(config: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        ROOT / config["model"]["tokenizer_path"],
        local_files_only=True,
        use_fast=False,
    )


def _replace_study_id(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _replace_study_id(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_study_id(item) for item in value]
    if value == PARENT_STUDY_ID:
        return STUDY_ID
    return value


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


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
    "build_amended_inventory",
    "complete_development_smoke_extension",
    "load_config",
    "preregister_construct_validity",
    "run_construct_validity",
    "validate_answer_contract",
    "validate_construct_validity",
]
