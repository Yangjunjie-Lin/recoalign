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
    rows: list[dict[str, Any]] = []
    for experiment in experiments["experiments"]:
        config = project / experiment["config"]
        manifest = project / experiment["dataset"]["manifest"]
        rows.append(
            {
                "experiment_id": experiment["experiment_id"],
                "kind": "evaluation",
                "final_config": experiment["config"],
                "config_sha256": sha256_file(config) if config.is_file() else None,
                "commit": commit,
                "checkpoint": None,
                "checkpoint_sha256": None,
                "dataset_version": experiment["dataset"]["version"],
                "dataset_manifest": experiment["dataset"]["manifest"],
                "dataset_manifest_sha256": sha256_file(manifest) if manifest.is_file() else None,
                "seeds": _config_seeds(config),
                "status": "protocol_registered_evidence_pending",
                "evidence_role": experiment["model"]["evidence_role"],
            }
        )
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
        "schema_version": 1,
        "freeze_name": "v2.0-paper-ready",
        "freeze_status": "candidate_not_released",
        "protocol_changes_allowed": False,
        "results_may_only_be_appended_under_frozen_protocol": True,
        "git": {
            "commit": commit,
            "dirty": dirty,
            "tag_created": False,
            "tag_blocker": (
                "Scientific submission decision is NO-GO and worktree is dirty; tagging would be misleading."
            ),
        },
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
