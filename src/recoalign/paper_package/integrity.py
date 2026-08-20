# ruff: noqa: E501
"""Final claim, leakage, and reproducibility gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recoalign.analysis.mechanistic.registry import validate_mechanistic_registry
from recoalign.evaluation.vlm_benchmark.matrix import load_matrix_config, validate_evaluation_matrix
from recoalign.research_registry import validate_research_registries
from recoalign.training.registry import validate_training_registry

from .common import project_root, write_yaml


def build_integrity_report(root: str | Path | None = None) -> dict[str, Any]:
    project = project_root(root)
    checks: dict[str, dict[str, Any]] = {}
    checks["claims"] = _check_claims(project)
    checks["leakage"] = _check_leakage(project)
    checks["reproducibility"] = _check_files(
        project,
        "reproducibility",
        [
            "environment.yml",
            "requirements.txt",
            "hardware.md",
            "dataset_protocol.md",
            "training_protocol.md",
            "evaluation_protocol.md",
        ],
    )
    checks["submission"] = _check_files(
        project,
        "submission",
        [
            "code_structure.md",
            "reproducibility_checklist.md",
            "artifact_description.md",
            "limitations.md",
        ],
    )
    checks["exports"] = _check_files(
        project,
        "reports",
        [
            "final_repository_audit.md",
            "results/main_results.tex",
            "latex/main_results.tex",
            "figures/figure_1_research_overview.svg",
        ],
    )
    blockers: list[str] = []
    if not checks["claims"]["passed"]:
        blockers.append("one or more mapped claims are missing or incorrectly marked")
    if not checks["leakage"]["passed"]:
        blockers.append("registry/leakage validation is incomplete")
    if not checks["reproducibility"]["passed"]:
        blockers.append("reproducibility package is incomplete")
    if not checks["submission"]["passed"] or not checks["exports"]["passed"]:
        blockers.append("paper-ready artifact files are incomplete")
    decision_path = project / "reports/decision_report.json"
    complete = 0
    planned = 432
    if decision_path.is_file():
        payload = json.loads(decision_path.read_text(encoding="utf-8"))
        complete = int(payload.get("complete_cells", 0))
        planned = int(payload.get("planned_cells", planned))
    if complete < planned:
        blockers.append(
            f"comprehensive benchmark matrix is incomplete ({complete}/{planned} cells)"
        )
    mechanism = project / "reports/mechanistic/decision_report.yaml"
    if mechanism.is_file():
        import yaml

        payload = yaml.safe_load(mechanism.read_text(encoding="utf-8")) or {}
        if payload.get("decision") != "GO" or payload.get("real_vlm_mechanistic_evidence") != "GO":
            blockers.append("real-VLM mechanistic evidence is not GO")
    else:
        blockers.append("mechanistic decision report is missing")
    report = {
        "schema_version": 1,
        "path": "reports/integrity_report.yaml",
        "engineering_artifacts_ready": all(check["passed"] for check in checks.values()),
        "scientific_submission_ready": not blockers,
        "checks": checks,
        "blockers": blockers,
        "decision": "GO" if not blockers else "NO-GO",
    }
    write_yaml(project / "reports/integrity_report.yaml", report)
    return report


def validate_paper_package(root: str | Path | None = None) -> dict[str, Any]:
    project = project_root(root)
    report_path = project / "reports/integrity_report.yaml"
    if not report_path.is_file():
        raise FileNotFoundError("reports/integrity_report.yaml is missing; build package first")
    import yaml

    report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("integrity report must be a mapping")
    if report.get("engineering_artifacts_ready") is not True:
        raise ValueError("paper package artifact gate failed")
    if report.get("decision") != "NO-GO" and report.get("scientific_submission_ready") is not True:
        raise ValueError("inconsistent paper package decision")
    return report


def _check_claims(project: Path) -> dict[str, Any]:
    path = project / "docs/evidence_map.yaml"
    if not path.is_file():
        return {"passed": False, "reason": "evidence map missing"}
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    claims = payload.get("claims", [])
    invalid = []
    for claim in claims:
        if claim.get("status") == "verified":
            if (
                not claim.get("artifact_inventory_complete")
                or claim.get("evidence_role") == "infrastructure_validation"
            ):
                invalid.append(claim.get("id"))
    return {
        "passed": not invalid and bool(claims),
        "claim_count": len(claims),
        "invalid_verified_claims": invalid,
    }


def _check_leakage(project: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    try:
        checks["research_registry"] = validate_research_registries(project)
        checks["benchmark_matrix"] = validate_evaluation_matrix(
            load_matrix_config(project / "configs/benchmarks/comprehensive_matrix.yaml")
        )
        checks["mechanistic_registry"] = validate_mechanistic_registry(
            project / "configs/ablations/mechanistic_registry.yaml"
        )
        checks["training_registry"] = validate_training_registry(
            project / "research/experiments/training_registry.yaml"
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        return {"passed": False, "error": str(exc), "checks": checks}
    matrix = checks["benchmark_matrix"]
    return {
        "passed": all(
            checks[name].get("valid") is True
            for name in (
                "research_registry",
                "benchmark_matrix",
                "mechanistic_registry",
                "training_registry",
            )
        )
        and matrix.get("no_selective_reporting") is True,
        "checks": checks,
        "leakage_assertions": {
            "protocol_lock": True,
            "test_information_leakage": "registered manifests and split protocols required",
            "no_selective_reporting": matrix.get("no_selective_reporting"),
        },
    }


def _check_files(project: Path, prefix: str, names: list[str]) -> dict[str, Any]:
    rows = [
        {"path": f"{prefix}/{name}", "exists": (project / prefix / name).is_file()}
        for name in names
    ]
    return {"passed": all(row["exists"] for row in rows), "files": rows}
