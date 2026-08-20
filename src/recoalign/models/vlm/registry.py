"""Backbone registry, configuration validation, and process-local model pooling."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from recoalign.models.vlm.base import BaseVLM, ReferenceVLM
from recoalign.models.vlm.cache import InferenceCache
from recoalign.models.vlm.evaluation import AnswerEvaluator
from recoalign.models.vlm.internvl import InternVLVLM
from recoalign.models.vlm.llava import Llava15VLM
from recoalign.models.vlm.llava_next import LlavaNextVLM
from recoalign.models.vlm.prompting import load_prompt_protocol
from recoalign.models.vlm.qwen_vl import QwenVLVLM

MODEL_CONFIGS = {
    "reference": "configs/models/reference.yaml",
    "llava_1_5_7b": "configs/models/llava_1_5_7b.yaml",
    "llava_next": "configs/models/llava_next.yaml",
    "qwen_vl": "configs/models/qwen_vl.yaml",
    "internvl": "configs/models/internvl.yaml",
}
ALIASES = {
    "llava": "llava_1_5_7b",
    "llava-1.5-7b": "llava_1_5_7b",
    "llava15": "llava_1_5_7b",
    "qwen2_vl": "qwen_vl",
}
ADAPTERS: dict[str, type[BaseVLM]] = {
    "reference": ReferenceVLM,
    "llava_1_5_7b": Llava15VLM,
    "llava_next": LlavaNextVLM,
    "qwen_vl": QwenVLVLM,
    "internvl": InternVLVLM,
}


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    path: Path
    payload: dict[str, Any]
    sha256: str

    @property
    def scientific_evidence(self) -> bool:
        return bool(self.payload.get("scientific_evidence", False))

    @property
    def compatibility(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.payload.get("compatibility", ()))

    def experiment_model_config(self) -> dict[str, Any]:
        model = deepcopy(self.payload["model"])
        model["backend"] = self.name
        model["generation"] = deepcopy(self.payload["generation"])
        runtime = self.payload["runtime"]
        model["batch_size"] = int(runtime.get("batch_size", 1))
        model["cache_enabled"] = bool(runtime.get("cache_enabled", True))
        model["cache_dir"] = runtime.get("cache_dir")
        model["prompt_protocol"] = self.payload["prompt_protocol"]
        model["checkpoint_manifest"] = self.payload.get("checkpoint_manifest")
        model["model_config"] = self.path.as_posix()
        model["model_config_sha256"] = self.sha256
        model["scientific_evidence"] = self.scientific_evidence
        return model


class ModelRegistry:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else Path(__file__).resolve().parents[4]
        self._instances: dict[str, BaseVLM] = {}

    def names(self) -> tuple[str, ...]:
        return tuple(MODEL_CONFIGS)

    def resolve_name(self, value: str) -> str:
        name = ALIASES.get(value, value)
        if name not in MODEL_CONFIGS:
            raise ValueError(f"unknown VLM model {value!r}; available: {', '.join(self.names())}")
        return name

    def definition(self, value: str) -> ModelDefinition:
        name = self.resolve_name(value)
        path = self.root / MODEL_CONFIGS[name]
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"model config must be a mapping: {path}")
        validate_model_definition(payload, expected_name=name)
        return ModelDefinition(
            name=name,
            path=path,
            payload=payload,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    def apply_to_experiment(
        self, experiment_config: dict[str, Any], model_name: str
    ) -> dict[str, Any]:
        definition = self.definition(model_name)
        experiment_id = str(experiment_config["experiment"]["id"])
        if experiment_id not in definition.compatibility:
            raise ValueError(
                f"{definition.name} is not registered as compatible with {experiment_id}"
            )
        resolved = deepcopy(experiment_config)
        model = definition.experiment_model_config()
        model["seed"] = int(resolved["experiment"]["seed"])
        resolved["model"] = model
        resolved.setdefault("vlm_evaluation", {}).update(
            {
                "model_name": definition.name,
                "model_config": definition.path.as_posix(),
                "model_config_sha256": definition.sha256,
            }
        )
        return resolved

    def get_or_create(self, model_config: dict[str, Any], *, seed: int) -> BaseVLM:
        key = _instance_key(model_config, seed)
        if key not in self._instances:
            self._instances[key] = _create_instance(model_config, seed=seed)
        return self._instances[key]

    def clear(self) -> None:
        self._instances.clear()


_REGISTRY = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    return _REGISTRY


def create_vlm(model_config: dict[str, Any], *, seed: int = 7) -> BaseVLM:
    return _REGISTRY.get_or_create(model_config, seed=seed)


def validate_model_definition(payload: dict[str, Any], *, expected_name: str | None = None) -> None:
    schema_path = Path(__file__).resolve().parents[4] / "schemas" / "vlm_model.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(payload, schema)
    required = {
        "schema_version",
        "name",
        "adapter",
        "scientific_evidence",
        "model",
        "generation",
        "runtime",
        "prompt_protocol",
        "compatibility",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"model config is missing required fields: {missing}")
    if payload["schema_version"] != 1:
        raise ValueError("model config schema_version must be 1")
    name = str(payload["name"])
    if expected_name is not None and name != expected_name:
        raise ValueError(f"model config name {name!r} does not match registry {expected_name!r}")
    if name not in ADAPTERS or payload["adapter"] != name:
        raise ValueError(f"model config adapter must identify a registered adapter: {name}")
    if not isinstance(payload["model"], dict):
        raise ValueError("model config model must be a mapping")
    if not isinstance(payload["generation"], dict):
        raise ValueError("model config generation must be a mapping")
    generation = payload["generation"]
    if bool(generation.get("do_sample", False)):
        raise ValueError("Phase 2.1 model evaluation requires deterministic do_sample=false")
    if float(generation.get("temperature", 0.0)) != 0.0:
        raise ValueError("Phase 2.1 model evaluation requires temperature=0")
    if int(generation.get("max_new_tokens", 0)) <= 0:
        raise ValueError("generation.max_new_tokens must be positive")
    if not isinstance(payload["runtime"], dict) or int(
        payload["runtime"].get("batch_size", 0)
    ) <= 0:
        raise ValueError("runtime.batch_size must be positive")
    compatibility = payload["compatibility"]
    if not isinstance(compatibility, list) or not compatibility:
        raise ValueError("model compatibility must be a non-empty experiment list")
    if any(value not in {"EXP001", "EXP002", "EXP003"} for value in compatibility):
        raise ValueError("model compatibility contains an unknown Phase-1 experiment")


def validate_inline_model_config(model: dict[str, Any]) -> None:
    backend = ALIASES.get(str(model.get("backend", "")), str(model.get("backend", "")))
    if backend not in ADAPTERS:
        raise ValueError(f"configuration model.backend must be one of {tuple(ADAPTERS)}")
    if backend != "reference" and not model.get("model_path", model.get("model_dir")):
        raise ValueError(f"{backend} configuration requires model_path")


def dry_run_model(model_config: dict[str, Any], *, seed: int = 7) -> dict[str, Any]:
    model = _create_instance(model_config, seed=seed)
    report = (
        model.dry_run()
        if hasattr(model, "dry_run")
        else {
            "adapter": model.model_id,
            "adapter_ready": True,
            "runtime_ready": True,
            "weights_loaded": False,
        }
    )
    return {**report, "provenance": model.provenance()}


def _create_instance(model_config: dict[str, Any], *, seed: int) -> BaseVLM:
    validate_inline_model_config(model_config)
    backend = ALIASES.get(str(model_config["backend"]), str(model_config["backend"]))
    prompt = load_prompt_protocol(model_config.get("prompt_protocol"))
    cache = InferenceCache(
        model_config.get("cache_dir"), enabled=bool(model_config.get("cache_enabled", False))
    )
    common = {
        "prompt_protocol": prompt,
        "evaluator": AnswerEvaluator(),
        "cache": cache,
        "batch_size": int(model_config.get("batch_size", 1)),
    }
    if backend == "reference":
        accuracy = {
            str(key): float(value)
            for key, value in dict(model_config.get("condition_accuracy", {})).items()
        }
        return ReferenceVLM(
            seed=int(model_config.get("seed", seed)),
            condition_accuracy=accuracy,
            **common,
        )
    normalized = dict(model_config)
    normalized["model_path"] = normalized.get("model_path", normalized.get("model_dir"))
    adapter = ADAPTERS[backend]
    return adapter.from_config(normalized, **common)  # type: ignore[attr-defined]


def _instance_key(model_config: dict[str, Any], seed: int) -> str:
    payload = deepcopy(model_config)
    backend = ALIASES.get(str(payload.get("backend")), str(payload.get("backend")))
    if backend == "reference":
        payload["seed"] = seed
    else:
        payload.pop("seed", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


__all__ = [
    "ModelDefinition",
    "ModelRegistry",
    "create_vlm",
    "dry_run_model",
    "get_model_registry",
    "validate_inline_model_config",
    "validate_model_definition",
]
