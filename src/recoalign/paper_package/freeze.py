# ruff: noqa: E501
"""Closed experiment inventory and guarded paper-tag creation."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .common import git_output, load_yaml, project_root, sha256_file, write_yaml

BASELINE_COMMIT = "7567d3a579384bcac76eb2b30dd6de538c62efe9"


class FreezeError(RuntimeError):
    """Raised when a paper freeze would misrepresent repository state."""


def build_frozen_registry(root: str | Path | None = None) -> dict[str, Any]:
    project = project_root(root)
    registered = load_yaml(project / "research/experiments/experiment_registry.yaml")
    paper_manifest = load_yaml(project / "reports/paper_evidence/artifact_manifest.yaml")
    paper_experiments = paper_manifest["experiments"]
    model = paper_manifest["model"]
    rows: list[dict[str, Any]] = []

    for experiment in registered["experiments"]:
        experiment_id = experiment["experiment_id"]
        evidence = paper_experiments[experiment_id]
        rows.append(
            {
                "experiment_id": experiment_id,
                "kind": "frozen_evaluation",
                "final_config": experiment["config"],
                "config_sha256": sha256_file(project / experiment["config"]),
                "model": model["identifier"],
                "model_revision": model["revision"],
                "checkpoint_fingerprint": model["checkpoint_fingerprint"],
                "status": evidence["status"],
                "decision": evidence["decision"],
                "claim_eligible": evidence["status"] == "complete" and evidence["decision"] in {"GO", "NO-GO"},
                "evidence_manifest": "reports/paper_evidence/artifact_manifest.yaml",
            }
        )

    rows.extend(
        [
            _closed_study(project, "PIVOT_EXP_A", "research/pivot_validation/config.yaml", "complete", "NO-GO", "research/pivot_validation/results/artifact_manifest.yaml"),
            _closed_study(project, "PIVOT_EXP_A2", "research/causal_separation/PIVOT_EXP_A2/config.yaml", "complete", "INCONCLUSIVE", "research/causal_separation/PIVOT_EXP_A2/results/artifact_manifest.yaml"),
            _closed_study(project, "PIVOT_EXP_A3", "research/construct_validity/PIVOT_EXP_A3/config.yaml", "blocked_preinference", "RUNTIME_BLOCKED_PREINFERENCE", "research/construct_validity/PIVOT_EXP_A3/v1_artifact_manifest.yaml"),
            _closed_study(project, "PIVOT_EXP_A3R", "research/construct_validity/PIVOT_EXP_A3R/config.yaml", "instrument_retired_before_validation", "SECONDARY_CONTRACT_RUNTIME_FAILURE", "research/construct_validity/PIVOT_EXP_A3R/final_artifact_manifest.yaml"),
            _closed_study(project, "PIVOT_EXP_A3P", "research/construct_validity/PIVOT_EXP_A3P/config.yaml", "complete", "SEMANTIC_MANIPULATION_FAILURE", "research/construct_validity/PIVOT_EXP_A3P/results/artifact_manifest.yaml"),
        ]
    )
    rows.extend(
        {
            "experiment_id": experiment_id,
            "kind": "historical_training_contract",
            "status": "inactive_retained_for_provenance",
            "decision": "NOT_AUTHORIZED",
            "claim_eligible": False,
        }
        for experiment_id in ("TRAIN001", "TRAIN002", "TRAIN003")
    )

    payload = {
        "schema_version": 3,
        "freeze_name": "recoalign-line-closeout-after-pivot-exp-a3p",
        "freeze_status": "research_line_closed",
        "evidence_baseline_commit": BASELINE_COMMIT,
        "protocol_changes_allowed": False,
        "frozen_results_modification_allowed": False,
        "scientific_submission_decision": "NO-GO",
        "program_decision": "TERMINATE_CURRENT_PROGRAM",
        "model_manifest": "reports/paper_evidence/model_manifest.yaml",
        "original_evidence_manifest": "reports/paper_evidence/artifact_manifest.yaml",
        "latest_evidence_manifest": "research/construct_validity/PIVOT_EXP_A3P/results/artifact_manifest.yaml",
        "experiments": rows,
        "inactive_future_work": {
            "recoalign_training": True,
            "comprehensive_432_cell_matrix": True,
            "mechanistic_method_evaluation": True,
        },
        "authorization": {
            "selected_candidate": "STOP",
            "new_preregistration_allowed": False,
            "real_inference_allowed": False,
            "independent_backbone_replication_allowed": False,
            "model_development_allowed": False,
            "paper_writing_allowed": False,
            "maximum_additional_diagnostic_pivots": 0,
            "archive_and_release_allowed": True,
        },
    }
    write_yaml(project / "experiments/frozen_registry.yaml", payload)
    return payload


def create_paper_tag(
    root: str | Path | None = None,
    *,
    tag: str = "v2.0-paper-ready",
    create: bool = False,
) -> dict[str, Any]:
    project = project_root(root)
    readiness_path = project / "reports/submission_readiness.yaml"
    if not readiness_path.is_file():
        raise FreezeError("submission readiness report is missing; build the package first")
    readiness = load_yaml(readiness_path)
    if readiness.get("scientific_submission_decision") != "GO":
        raise FreezeError("paper tag blocked: scientific submission decision is not GO")
    if git_output(project, "status", "--porcelain=v1", "--untracked-files=all"):
        raise FreezeError("paper tag blocked: worktree is dirty")
    commit = git_output(project, "rev-parse", "HEAD")
    if git_output(project, "tag", "--list", tag):
        raise FreezeError(f"paper tag already exists: {tag}")
    result = {"tag": tag, "commit": commit, "eligible": True, "created": False}
    if create:
        subprocess.run(["git", "tag", "-a", tag, "-m", "ReCoAlign paper-ready evidence freeze"], cwd=project, check=True, timeout=30)
        result["created"] = True
    return result


def _closed_study(
    project: Path,
    experiment_id: str,
    config: str,
    status: str,
    decision: str,
    evidence_manifest: str,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "kind": "frozen_diagnostic",
        "final_config": config,
        "config_sha256": sha256_file(project / config),
        "status": status,
        "decision": decision,
        "claim_eligible": experiment_id == "PIVOT_EXP_A3P" and status == "complete",
        "evidence_manifest": evidence_manifest,
    }
