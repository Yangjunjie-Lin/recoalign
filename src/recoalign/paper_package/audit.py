# ruff: noqa: E501
"""Repository and artifact audit for submission preparation."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from recoalign.research_registry import load_experiment_registry

from .common import git_output, project_root, write_text

_TEMP_NAMES = ("smoke", "check", "dryrun", "debug", "tmp", "temp")
_LOCAL_PATH = re.compile(r"(?:[A-Za-z]:\\Users\\|[A-Za-z]:\\UoM|/home/|/Users/)")
_SECRET = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)\s*[:=]\s*['\"][^'\"]+"
)


def audit_repository(root: str | Path | None = None) -> dict[str, Any]:
    project = project_root(root)
    tracked = _tracked_files(project)
    python_files = [
        path for path in tracked if path.suffix == ".py" and "archive" not in path.parts
    ]
    duplicate_groups = _duplicate_groups(python_files, project)
    local_paths, potential_secrets = _scan_sensitive(project, tracked)
    experiment_rows = _audit_experiments(project)
    output_dirs = project / "outputs"
    temporary_outputs = []
    if output_dirs.is_dir():
        temporary_outputs = sorted(
            path.relative_to(project).as_posix()
            for path in output_dirs.iterdir()
            if path.is_dir() and any(token in path.name.lower() for token in _TEMP_NAMES)
        )

    report = {
        "schema_version": 1,
        "code": {
            "tracked_python_files": len(python_files),
            "duplicate_content_groups": duplicate_groups,
            "compatibility_wrapper_note": (
                "Top-level models/analysis/evaluation modules may intentionally wrap src/recoalign; "
                "duplicate candidates require human review before deletion."
            ),
            "automatic_dead_code_deletion": False,
        },
        "experiments": experiment_rows,
        "documentation": _documentation_audit(project),
        "release_hygiene": {
            "temporary_or_validation_output_candidates": temporary_outputs,
            "temporary_outputs_deleted": False,
            "reason_not_deleted": "Ignored outputs can contain retained failed runs and user evidence.",
            "tracked_local_path_findings": local_paths,
            "potential_secret_findings": potential_secrets,
        },
        "git": {
            "commit": git_output(project, "rev-parse", "HEAD"),
            "branch": git_output(project, "branch", "--show-current"),
            "dirty": bool(git_output(project, "status", "--porcelain=v1", "--untracked-files=all")),
        },
    }
    write_text(project / "reports/final_repository_audit.md", _render_audit(report))
    return report


def _tracked_files(project: Path) -> list[Path]:
    output = git_output(project, "ls-files", "--cached", "--others", "--exclude-standard") or ""
    return [project / line for line in output.splitlines() if (project / line).is_file()]


def _duplicate_groups(paths: list[Path], project: Path) -> list[list[str]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        groups[digest].append(path)
    return [
        sorted(path.relative_to(project).as_posix() for path in values)
        for values in groups.values()
        if len(values) > 1
    ]


def _scan_sensitive(project: Path, paths: list[Path]) -> tuple[list[str], list[str]]:
    local: list[str] = []
    secrets: list[str] = []
    for path in paths:
        if path.suffix.lower() not in {".py", ".md", ".yaml", ".yml", ".json", ".toml", ".txt"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(project).as_posix()
        if relative == "src/recoalign/paper_package/audit.py":
            continue
        if _LOCAL_PATH.search(text):
            local.append(relative)
        if _SECRET.search(text):
            secrets.append(relative)
    return sorted(set(local)), sorted(set(secrets))


def _audit_experiments(project: Path) -> list[dict[str, Any]]:
    registry = load_experiment_registry(project)
    rows: list[dict[str, Any]] = []
    for experiment in registry["experiments"]:
        identifier = experiment["experiment_id"]
        output = project / "outputs" / identifier
        rows.append(
            {
                "experiment_id": identifier,
                "registered_status": experiment["status"],
                "evidence_role": experiment["model"]["evidence_role"],
                "config": {
                    "path": experiment["config"],
                    "exists": (project / experiment["config"]).is_file(),
                },
                "dataset_manifest": {
                    "path": experiment["dataset"]["manifest"],
                    "exists": (project / experiment["dataset"]["manifest"]).is_file(),
                },
                "metrics_files": len(list(output.glob("**/metrics.json")))
                if output.exists()
                else 0,
                "prediction_files": len(list(output.glob("**/predictions.jsonl")))
                if output.exists()
                else 0,
                "claim_eligible": experiment["model"]["scientific_decision_allowed"],
            }
        )
    return rows


def _documentation_audit(project: Path) -> dict[str, Any]:
    required = [
        "docs/experiment_governance.md",
        "docs/vlm_integration.md",
        "docs/interface_diagnosis.md",
        "docs/method_design.md",
        "docs/training_framework.md",
        "docs/comprehensive_evaluation.md",
        "docs/mechanistic_analysis.md",
    ]
    return {
        "required": [{"path": path, "exists": (project / path).is_file()} for path in required],
        "complete": all((project / path).is_file() for path in required),
    }


def _render_audit(report: dict[str, Any]) -> str:
    experiments = report["experiments"]
    lines = [
        "# Final repository audit",
        "",
        "## Outcome",
        "",
        "The research infrastructure is auditable, but the repository is not scientifically "
        "submission-ready because real-VLM and comprehensive matrix evidence is incomplete.",
        "",
        "## Experiment artifacts",
        "",
        "| Experiment | Role | Config | Manifest | Metrics | Predictions | Claim eligible |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in experiments:
        lines.append(
            f"| {row['experiment_id']} | {row['evidence_role']} | "
            f"{row['config']['exists']} | {row['dataset_manifest']['exists']} | "
            f"{row['metrics_files']} | {row['prediction_files']} | {row['claim_eligible']} |"
        )
    hygiene = report["release_hygiene"]
    lines.extend(
        [
            "",
            "## Code and release hygiene",
            "",
            f"- Tracked Python files: {report['code']['tracked_python_files']}",
            f"- Duplicate-content groups requiring review: {len(report['code']['duplicate_content_groups'])}",
            f"- Temporary/validation output candidates retained: {len(hygiene['temporary_or_validation_output_candidates'])}",
            f"- Tracked local-path findings: {len(hygiene['tracked_local_path_findings'])}",
            f"- Potential secret findings: {len(hygiene['potential_secret_findings'])}",
            f"- Dirty worktree: {report['git']['dirty']}",
            "",
            "Ignored experiment outputs were not deleted: failed and incomplete runs are scientific "
            "audit evidence, and some may be user-owned artifacts.",
        ]
    )
    return "\n".join(lines)
