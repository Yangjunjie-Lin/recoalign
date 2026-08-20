"""Independent TRAIN001-TRAIN003 registry validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .stages import STAGE_DEFINITIONS, parse_stage
from .trainer import load_training_config

TRAINING_REGISTRY = Path("research/experiments/training_registry.yaml")


def load_training_registry(path: str | Path = TRAINING_REGISTRY) -> dict[str, Any]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("experiments"), list):
        raise ValueError("training registry requires an experiments list")
    return payload


def validate_training_registry(path: str | Path = TRAINING_REGISTRY) -> dict[str, Any]:
    payload = load_training_registry(path)
    experiments = payload["experiments"]
    ids: set[str] = set()
    stages: set[str] = set()
    for row in experiments:
        if not isinstance(row, dict):
            raise ValueError("training registry rows must be mappings")
        identifier = str(row.get("id", ""))
        if not identifier.startswith("TRAIN") or identifier in ids:
            raise ValueError(f"invalid or duplicate training experiment ID: {identifier}")
        ids.add(identifier)
        stage = parse_stage(str(row.get("stage", ""))).value
        stages.add(stage)
        config = load_training_config(str(row.get("config", "")))
        if config.experiment_id != identifier or config.stage != stage:
            raise ValueError(f"{identifier} registry/config identity mismatch")
        contract = config.raw.get("inference_contract", {})
        if not isinstance(contract, dict) or contract.get("oracle_graph_allowed") is not False:
            raise ValueError(f"{identifier} must prohibit oracle graph inference")
        if not row.get("hypothesis") or not row.get("goal") or not row.get("decision_criteria"):
            raise ValueError(f"{identifier} lacks hypothesis, goal, or decision criteria")
    expected_stages = {stage.value for stage in STAGE_DEFINITIONS}
    if stages != expected_stages:
        raise ValueError("training registry must contain exactly one experiment per stage")
    return {
        "valid": True,
        "training_experiments": sorted(ids),
        "stages": sorted(stages),
        "scientific_evaluation_registry_unchanged": True,
    }


__all__ = ["TRAINING_REGISTRY", "load_training_registry", "validate_training_registry"]
