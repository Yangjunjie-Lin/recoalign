"""Configuration loading, validation, and deterministic hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is malformed."""


_REQUIRED_PATHS = (
    "experiment.name",
    "experiment.seed",
    "experiment.output_dir",
    "model.framework",
    "model.name",
    "model.pretrained",
    "model.manifest",
    "data.dataset",
    "data.root",
    "data.manifest",
    "data.split",
    "evaluation.recall_at",
    "training.enabled",
)

_STRING_PATHS = (
    "experiment.name",
    "experiment.output_dir",
    "model.framework",
    "model.name",
    "model.pretrained",
    "model.manifest",
    "data.dataset",
    "data.root",
    "data.manifest",
    "data.split",
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML experiment configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ConfigError("configuration root must be a mapping")
    validate_config(payload)
    return payload


def validate_config(config: dict[str, Any]) -> None:
    """Validate the stable research configuration contract."""
    if not isinstance(config, dict):
        raise ConfigError("configuration root must be a mapping")

    missing = [path for path in _REQUIRED_PATHS if _lookup(config, path) is None]
    if missing:
        raise ConfigError(f"missing required configuration fields: {', '.join(missing)}")

    seed = _lookup(config, "experiment.seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigError("experiment.seed must be a non-negative integer")

    recall_at = _lookup(config, "evaluation.recall_at")
    if (
        not isinstance(recall_at, list)
        or not recall_at
        or any(not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in recall_at)
    ):
        raise ConfigError("evaluation.recall_at must be a non-empty list of positive integers")
    if len(set(recall_at)) != len(recall_at):
        raise ConfigError("evaluation.recall_at must not contain duplicates")

    if not isinstance(_lookup(config, "training.enabled"), bool):
        raise ConfigError("training.enabled must be a boolean")

    for path in _STRING_PATHS:
        value = _lookup(config, path)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{path} must be a non-empty string")


def config_digest(config: dict[str, Any]) -> str:
    """Return a stable SHA-256 digest for a validated configuration."""
    validate_config(config)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_digest(path: str | Path) -> str:
    """Return the SHA-256 digest of a configuration or manifest file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _lookup(config: dict[str, Any], dotted_path: str) -> Any:
    value: Any = config
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value
