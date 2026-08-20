"""Load and cross-validate the hypothesis and experiment registries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from recoalign.schema_validation import repository_root, validate_payload

HYPOTHESIS_REGISTRY = Path("research/hypotheses/hypothesis_registry.yaml")
EXPERIMENT_REGISTRY = Path("research/experiments/experiment_registry.yaml")
PROTOCOL_TEMPLATE = Path("research/protocols/experiment_template.md")
REQUIRED_PROTOCOL_HEADINGS = (
    "## Scientific Question",
    "## Hypothesis",
    "## Variables",
    "## Controlled Factors",
    "## Evaluation Metrics",
    "## Statistical Protocol",
    "## GO / NO-GO Criteria",
)


class RegistryError(ValueError):
    """Raised when scientific registrations are incomplete or inconsistent."""


def load_hypothesis_registry(root: str | Path | None = None) -> dict[str, Any]:
    project = Path(root).resolve() if root else repository_root()
    payload = _load_yaml(project / HYPOTHESIS_REGISTRY)
    validate_payload("hypothesis_registry", payload)
    return payload


def load_experiment_registry(root: str | Path | None = None) -> dict[str, Any]:
    project = Path(root).resolve() if root else repository_root()
    payload = _load_yaml(project / EXPERIMENT_REGISTRY)
    validate_payload("experiment_registry", payload)
    return payload


def validate_research_registries(root: str | Path | None = None) -> dict[str, Any]:
    """Validate schemas, IDs, cross-references, protocols, configs, and no-training gates."""

    project = Path(root).resolve() if root else repository_root()
    hypotheses_payload = load_hypothesis_registry(project)
    experiments_payload = load_experiment_registry(project)
    hypotheses = _unique_by(hypotheses_payload["hypotheses"], "id")
    experiments = _unique_by(experiments_payload["experiments"], "experiment_id")
    _validate_protocol(project / PROTOCOL_TEMPLATE)

    for hypothesis_id, hypothesis in hypotheses.items():
        required = hypothesis["required_experiment"]
        if required not in experiments:
            raise RegistryError(f"{hypothesis_id} requires unregistered experiment {required}")
        if experiments[required]["hypothesis_id"] != hypothesis_id:
            raise RegistryError(f"{required} does not link back to {hypothesis_id}")

    from experiments.runtime import load_config

    for experiment_id, experiment in experiments.items():
        hypothesis_id = experiment["hypothesis_id"]
        if hypothesis_id not in hypotheses:
            raise RegistryError(f"{experiment_id} references unknown hypothesis {hypothesis_id}")
        protocol = project / experiment["protocol"]
        config_path = project / experiment["config"]
        manifest_path = project / experiment["dataset"]["manifest"]
        for label, path in (
            ("protocol", protocol),
            ("config", config_path),
            ("dataset manifest", manifest_path),
        ):
            if not path.is_file():
                raise RegistryError(f"{experiment_id} {label} does not exist: {path}")
        _validate_protocol(protocol, hypothesis_id=hypothesis_id)

        config = load_config(config_path)
        config_experiment = config["experiment"]
        if config_experiment.get("id") != experiment_id:
            raise RegistryError(f"{experiment_id} does not match config experiment.id")
        if config_experiment.get("hypothesis_id") != hypothesis_id:
            raise RegistryError(f"{experiment_id} config does not link to {hypothesis_id}")
        conditions = config.get("evaluation", {}).get("conditions")
        if conditions != experiment["input_conditions"]:
            raise RegistryError(f"{experiment_id} config conditions differ from its registration")
        configured_data = config.get("data", {})
        if configured_data.get("dataset") != experiment["dataset"]["name"]:
            raise RegistryError(f"{experiment_id} config dataset differs from its registration")
        if configured_data.get("version") != experiment["dataset"]["version"]:
            raise RegistryError(f"{experiment_id} config dataset version is not registered")
        if config["model"].get("backend") != experiment["model"]["configured_backend"]:
            raise RegistryError(f"{experiment_id} config model backend is not registered")
        dataset_manifest = _load_yaml(manifest_path)
        validate_payload("dataset_manifest", dataset_manifest)
        if dataset_manifest.get("name") != experiment["dataset"]["name"]:
            raise RegistryError(f"{experiment_id} dataset manifest name is inconsistent")
        if dataset_manifest.get("version") != experiment["dataset"]["version"]:
            raise RegistryError(f"{experiment_id} dataset manifest version is inconsistent")
        seeds = config_experiment.get("seeds")
        minimum_seeds = experiment["decision_rule"]["robustness"]["minimum_seeds"]
        if (
            not isinstance(seeds, list)
            or len(seeds) < minimum_seeds
            or len(set(seeds)) != len(seeds)
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                for seed in seeds
            )
        ):
            raise RegistryError(
                f"{experiment_id} requires at least {minimum_seeds} unique non-negative seeds"
            )
        if config.get("training", {}).get("enabled") is not False:
            raise RegistryError(f"{experiment_id} must set training.enabled=false")
        if experiment["model"]["training_allowed"] is not False:
            raise RegistryError(f"{experiment_id} must prohibit training")
        evidence_role = experiment["model"]["evidence_role"]
        decision_allowed = experiment["model"]["scientific_decision_allowed"]
        if decision_allowed != (evidence_role == "scientific_evidence"):
            raise RegistryError(
                f"{experiment_id} model evidence role and decision eligibility are inconsistent"
            )

    return {
        "hypotheses": len(hypotheses),
        "experiments": len(experiments),
        "hypothesis_ids": sorted(hypotheses),
        "experiment_ids": sorted(experiments),
        "valid": True,
    }


def get_registered_experiment(
    experiment_id: str, root: str | Path | None = None
) -> dict[str, Any]:
    registry = load_experiment_registry(root)
    for experiment in registry["experiments"]:
        if experiment["experiment_id"] == experiment_id:
            return experiment
    raise RegistryError(f"experiment is not registered: {experiment_id}")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"registry does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RegistryError(f"registry root must be a mapping: {path}")
    return payload


def _unique_by(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row[field]
        if value in indexed:
            raise RegistryError(f"duplicate {field}: {value}")
        indexed[value] = row
    return indexed


def _validate_protocol(path: Path, *, hypothesis_id: str | None = None) -> None:
    if not path.is_file():
        raise RegistryError(f"protocol does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    headings = {line.strip() for line in text.splitlines() if line.startswith("## ")}
    missing = [heading for heading in REQUIRED_PROTOCOL_HEADINGS if heading not in headings]
    if missing:
        raise RegistryError(
            f"protocol {path} is missing required sections: {', '.join(missing)}"
        )
    if hypothesis_id is not None and hypothesis_id not in text:
        raise RegistryError(f"protocol {path} does not identify hypothesis {hypothesis_id}")
