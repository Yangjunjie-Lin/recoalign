# ruff: noqa: E501
"""Immutable experiment inventory and guarded paper-tag creation."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import yaml

from .common import git_output, load_yaml, project_root, sha256_file, write_yaml


class FreezeError(RuntimeError):
    """Raised when a paper freeze would misrepresent repository state."""


def build_frozen_registry(root: str | Path | None = None) -> dict[str, Any]:
    project = project_root(root)
    experiments = load_yaml(project / "research/experiments/experiment_registry.yaml")
    training = load_yaml(project / "research/experiments/training_registry.yaml")
    commit = git_output(project, "rev-parse", "HEAD")
    dirty = bool(git_output(project, "status", "--porcelain=v1", "--untracked-files=all"))
    evidence_manifest_path = project / "reports/paper_evidence/artifact_manifest.yaml"
    evidence_manifest = load_yaml(evidence_manifest_path) if evidence_manifest_path.is_file() else {}
    evidence_experiments = evidence_manifest.get("experiments", {})
    model = evidence_manifest.get("model", {})
    rows: list[dict[str, Any]] = []
    for experiment in experiments["experiments"]:
        config = project / experiment["config"]
        manifest = project / experiment["dataset"]["manifest"]
        evidence = evidence_experiments.get(experiment["experiment_id"])
        row = {
                "experiment_id": experiment["experiment_id"],
                "kind": "evaluation",
                "final_config": experiment["config"],
                "config_sha256": sha256_file(config) if config.is_file() else None,
                "commit": commit,
                "checkpoint": (
                    "manifests/checkpoints/llava_v1_5_7b.yaml" if evidence else None
                ),
                "checkpoint_sha256": (
                    sha256_file(project / "manifests/checkpoints/llava_v1_5_7b.yaml")
                    if evidence
                    else None
                ),
                "checkpoint_fingerprint": model.get("checkpoint_fingerprint") if evidence else None,
                "model": model.get("identifier") if evidence else None,
                "model_revision": model.get("revision") if evidence else None,
                "model_config": "configs/models/llava_1_5_7b.yaml" if evidence else None,
                "model_config_sha256": (
                    sha256_file(project / "configs/models/llava_1_5_7b.yaml")
                    if evidence
                    else None
                ),
                "dataset_version": experiment["dataset"]["version"],
                "dataset_manifest": experiment["dataset"]["manifest"],
                "dataset_manifest_sha256": sha256_file(manifest) if manifest.is_file() else None,
                "seeds": _config_seeds(config),
                "status": evidence.get("status") if evidence else "protocol_registered_evidence_pending",
                "decision": evidence.get("decision") if evidence else "PENDING",
                "evidence_role": "scientific_evidence" if evidence else experiment["model"]["evidence_role"],
                "claim_eligible": bool(
                    evidence
                    and evidence.get("status") == "complete"
                    and evidence.get("decision") in {"GO", "NO-GO"}
                ),
                "evidence_manifest": (
                    "reports/paper_evidence/artifact_manifest.yaml" if evidence else None
                ),
            }
        rows.append(row)
    for experiment in training["experiments"]:
        config = project / experiment["config"]
        payload = load_yaml(config)
        rows.append(
            {
                "experiment_id": experiment["id"],
                "kind": "training",
                "final_config": experiment["config"],
                "config_sha256": sha256_file(config),
                "commit": commit,
                "checkpoint": None,
                "checkpoint_sha256": None,
                "dataset_version": payload.get("dataset", {}).get("version"),
                "dataset_manifest": payload.get("dataset", {}).get("manifest"),
                "seeds": [payload.get("seed")],
                "status": (
                    "toy_validation_complete_real_training_pending"
                    if experiment["id"] != "TRAIN003"
                    else "pending_real_dataset_and_checkpoint"
                ),
            }
        )
    payload = {
        "schema_version": 2,
        "freeze_name": "llava-claim-evidence-20260820",
        "freeze_status": (
            "evidence_frozen_no_go" if evidence_manifest.get("scientific_decision") == "NO-GO"
            else "candidate_not_released"
        ),
        "protocol_changes_allowed": False,
        "results_may_only_be_appended_under_frozen_protocol": True,
        "git": {
            "commit": commit,
            "evidence_execution_commit": evidence_manifest.get("source_commit"),
            "dirty": dirty,
            "tag_created": False,
            "tag_blocker": (
                "Scientific submission decision is NO-GO; tagging would be misleading."
            ),
        },
        "model_manifest": (
            "reports/paper_evidence/model_manifest.yaml"
            if (project / "reports/paper_evidence/model_manifest.yaml").is_file()
            else None
        ),
        "evidence_artifact_manifest": (
            "reports/paper_evidence/artifact_manifest.yaml" if evidence_manifest else None
        ),
        "scientific_decision": evidence_manifest.get("scientific_decision", "PENDING"),
        "experiments": rows,
    }
    write_yaml(project / "experiments/frozen_registry.yaml", payload)
    return payload


def create_paper_tag(
    root: str | Path | None = None,
    *,
    tag: str = "v2.0-paper-ready",
    create: bool = False,
) -> dict[str, Any]:
    """Validate tag gates and optionally create an annotated tag.

    Tagging is intentionally impossible while the readiness decision is not GO, the
    worktree is dirty, or the frozen registry does not identify the current commit.
    """

    project = project_root(root)
    readiness_path = project / "reports/submission_readiness.yaml"
    if not readiness_path.is_file():
        raise FreezeError("submission readiness report is missing; build the package first")
    readiness = load_yaml(readiness_path)
    if readiness.get("scientific_submission_decision") != "GO":
        raise FreezeError("paper tag blocked: scientific submission decision is not GO")
    if git_output(project, "status", "--porcelain=v1", "--untracked-files=all"):
        raise FreezeError("paper tag blocked: worktree is dirty")
    registry = load_yaml(project / "experiments/frozen_registry.yaml")
    commit = git_output(project, "rev-parse", "HEAD")
    if registry.get("git", {}).get("commit") != commit:
        raise FreezeError("paper tag blocked: frozen registry commit does not match HEAD")
    if git_output(project, "tag", "--list", tag):
        raise FreezeError(f"paper tag already exists: {tag}")
    result = {"tag": tag, "commit": commit, "eligible": True, "created": False}
    if create:
        subprocess.run(
            ["git", "tag", "-a", tag, "-m", "ReCoAlign paper-ready evidence freeze"],
            cwd=project,
            check=True,
            timeout=30,
        )
        result["created"] = True
    return result


def _config_seeds(path: Path) -> list[int]:
    if not path.is_file():
        return []
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    experiment = payload.get("experiment", {})
    seeds = (
        experiment.get("seeds", payload.get("seeds", [])) if isinstance(experiment, dict) else []
    )
    return [int(seed) for seed in seeds] if isinstance(seeds, list) else []
