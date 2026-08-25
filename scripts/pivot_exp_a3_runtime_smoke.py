"""Run the PIVOT_EXP_A3 development-only CUDA/NF4 runtime smoke.

This utility never reads the frozen validation trial inventory and never writes predictions.
It exercises four preregistered response/scaffold paths on the separately registered development
seed, then emits a runtime-only report under the A3 validation directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from datasets.records import SceneRecord
from recoalign.construct_validity.answer_contract import (
    REGISTERED_CHOICE_IDS,
    parse_final_choice,
)
from recoalign.construct_validity.choice_scoring import (
    LlavaConditionalLikelihoodBackend,
    score_choices,
)
from recoalign.construct_validity.runtime import configure_frozen_tokenizer_runtime
from recoalign.construct_validity.trial_builder import build_validation_trials
from recoalign.models.vlm.cache import InferenceCache
from recoalign.models.vlm.registry import ModelRegistry

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "research/construct_validity/PIVOT_EXP_A3"
OUTPUT = ROOT / "outputs/construct_validity/PIVOT_EXP_A3"
DEVELOPMENT = OUTPUT / "inventory/development/dataset.jsonl"
M2_DEVELOPMENT = OUTPUT / "inventory/m2_development"
NEUTRAL_IMAGE = STUDY / "semantic_manipulations/neutral.png"
REPORT = STUDY / "validation/runtime_smoke_report.yaml"
DEVELOPMENT_SEED = 20260830


def _read_records() -> list[SceneRecord]:
    rows = [
        json.loads(line)
        for line in DEVELOPMENT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [SceneRecord.from_dict(row) for row in rows]
    if not records or any(
        int(record.metadata.get("seed", -1)) != DEVELOPMENT_SEED for record in records
    ):
        raise RuntimeError("runtime smoke input is not exclusively the frozen development seed")
    return records


def _m2_mapping(records: list[SceneRecord]) -> dict[tuple[str, str, str], str]:
    mapping: dict[tuple[str, str, str], str] = {}
    for record in records:
        for truth in ("oracle", "corrupted"):
            for context in ("neutral_image", "original_scene_image"):
                path = M2_DEVELOPMENT / record.scene_id / truth / f"{context}.png"
                if not path.is_file():
                    raise FileNotFoundError(path)
                mapping[(record.scene_id, truth, context)] = str(path.resolve())
    return mapping


def _select(trials: list[Any], **criteria: str) -> Any:
    matches = [
        trial
        for trial in trials
        if all(str(getattr(trial, field)) == value for field, value in criteria.items())
    ]
    if not matches:
        raise RuntimeError(f"no development smoke trial matches {criteria}")
    return sorted(matches, key=lambda trial: trial.key)[0]


def _score(backend: LlavaConditionalLikelihoodBackend, trial: Any) -> dict[str, Any]:
    first = score_choices(trial.prepared_input(), REGISTERED_CHOICE_IDS, backend=backend)
    second = score_choices(trial.prepared_input(), REGISTERED_CHOICE_IDS, backend=backend)
    finite = len(first.log_likelihoods) == 4 and all(
        math.isfinite(value) for value in first.log_likelihoods.values()
    )
    return {
        "trial_key": trial.key,
        "scene_id": trial.scene_id,
        "manipulation": trial.manipulation,
        "response_method": trial.response_method,
        "task": trial.task,
        "choice_scores": first.log_likelihoods,
        "choice_score_count": len(first.log_likelihoods),
        "choice_scores_finite": finite,
        "predicted_choice_id": first.predicted_choice_id,
        "deterministic_repeat_stable": first.to_dict() == second.to_dict(),
    }


def _generate(model: Any, trial: Any) -> dict[str, Any]:
    kwargs = {
        "temperature": 0.0,
        "do_sample": False,
        "max_new_tokens": 16,
        "num_beams": 1,
    }
    first = model.generate(trial.prompt, image=trial.image, **kwargs)
    second = model.generate(trial.prompt, image=trial.image, **kwargs)
    parsed = parse_final_choice(first)
    return {
        "trial_key": trial.key,
        "scene_id": trial.scene_id,
        "manipulation": trial.manipulation,
        "response_method": trial.response_method,
        "task": trial.task,
        "raw_output": first,
        "parseable": parsed.valid,
        "parsed_choice_id": parsed.choice_id,
        "parse_failure_reason": parsed.failure_reason,
        "deterministic_repeat_stable": first == second,
    }


def _atomic_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def main() -> int:
    expected_allocator = "max_split_size_mb:64,garbage_collection_threshold:0.70"
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") != expected_allocator:
        raise RuntimeError("frozen allocator environment is not active")
    if not torch.cuda.is_available():
        raise RuntimeError("development runtime smoke requires CUDA")
    tokenizer_runtime = configure_frozen_tokenizer_runtime()

    records = _read_records()
    trials = build_validation_trials(
        records,
        seed=DEVELOPMENT_SEED,
        neutral_image=NEUTRAL_IMAGE,
        m2_images=_m2_mapping(records),
    )
    selected = {
        "forced_choice": _select(
            trials,
            manipulation="M0",
            evidence_truth="oracle",
            image_context="neutral_image",
            response_method="forced_choice",
            task="shape",
        ),
        "free_generation": _select(
            trials,
            manipulation="M1",
            evidence_truth="oracle",
            image_context="neutral_image",
            response_method="free_generation",
            task="answer_contract_comprehension",
        ),
        "text_table": _select(
            trials,
            manipulation="M1",
            evidence_truth="oracle",
            image_context="neutral_image",
            response_method="forced_choice",
            task="color",
        ),
        "visual_legend": _select(
            trials,
            manipulation="M2",
            evidence_truth="oracle",
            image_context="neutral_image",
            response_method="forced_choice",
            task="object_identity",
        ),
    }

    registry = ModelRegistry(ROOT)
    definition = registry.definition("llava_1_5_7b")
    model = registry.get_or_create(definition.experiment_model_config(), seed=DEVELOPMENT_SEED)
    model.ensure_loaded()
    backend = model._require_backend()  # noqa: SLF001 - runtime audit boundary
    likelihood = LlavaConditionalLikelihoodBackend(model)

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
    model_load = {
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
        "no_cpu_fallback": all(
            "cpu" not in device for device in language_devices + vision_devices + projector_devices
        ),
    }

    cache = InferenceCache(OUTPUT / "runtime_smoke/cache", enabled=True)
    cache_payload = {
        "study": "PIVOT_EXP_A3",
        "role": "runtime_smoke_only",
        "development_seed": DEVELOPMENT_SEED,
    }
    cache_key = cache.key(cache_payload)
    cache.put(cache_key, "runtime_smoke_only", cache_payload)
    cache_path = cache._path(cache_key)  # noqa: SLF001 - explicit cache integrity audit
    cache_ok = cache.get(cache_key) == "runtime_smoke_only" and bool(
        cache_path and cache_path.is_file()
    )

    reports = {
        "forced_choice": _score(likelihood, selected["forced_choice"]),
        "free_generation": _generate(model, selected["free_generation"]),
        "text_table": _score(likelihood, selected["text_table"]),
        "visual_legend": _score(likelihood, selected["visual_legend"]),
    }
    checks = {
        "cuda_available": torch.cuda.is_available(),
        "no_oom": True,
        "no_cpu_fallback": model_load["no_cpu_fallback"],
        "weights_loaded": model_load["weights_loaded"],
        "vision_tower_loaded": model_load["vision_tower_loaded"],
        "projector_loaded": model_load["projector_loaded"],
        "nf4_loaded": model_load["is_loaded_in_4bit"] and four_bit_modules > 0,
        "exactly_four_choice_scores": all(
            reports[name]["choice_score_count"] == 4
            for name in ("forced_choice", "text_table", "visual_legend")
        ),
        "choice_scores_finite": all(
            reports[name]["choice_scores_finite"]
            for name in ("forced_choice", "text_table", "visual_legend")
        ),
        "free_generation_parseable": reports["free_generation"]["parseable"],
        "deterministic_repeat_stable": all(
            report["deterministic_repeat_stable"] for report in reports.values()
        ),
        "cache_writable_and_readable": cache_ok,
        "development_seed_only": all(trial.seed == DEVELOPMENT_SEED for trial in selected.values()),
        "validation_inventory_read": False,
        "predictions_written": False,
        "scientific_metrics_computed": False,
    }
    passed = all(
        value
        for key, value in checks.items()
        if key
        not in {"validation_inventory_read", "predictions_written", "scientific_metrics_computed"}
    ) and not any(
        checks[key]
        for key in (
            "validation_inventory_read",
            "predictions_written",
            "scientific_metrics_computed",
        )
    )
    report = {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A3",
        "status": "runtime_smoke_only",
        "passed": passed,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_built_cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0),
        "allocator_environment": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
        "tokenizer_runtime": tokenizer_runtime,
        "development_seed": DEVELOPMENT_SEED,
        "development_scene_ids": sorted({trial.scene_id for trial in selected.values()}),
        "model_load": model_load,
        "checks": checks,
        "trials": reports,
        "cache": {
            "key": cache_key,
            "path": str(cache_path.resolve()) if cache_path else None,
            "sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest() if cache_path else None,
            "separate_from_predictions": True,
        },
        "cuda_memory": {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    }
    _atomic_yaml(REPORT, report)
    print(
        json.dumps({"status": report["status"], "passed": passed, "report": str(REPORT)}, indent=2)
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
