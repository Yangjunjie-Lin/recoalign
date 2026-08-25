# ruff: noqa: E501
"""Build every paper-ready artifact in one deterministic, auditable operation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from recoalign.reproducibility import collect_environment

from .audit import audit_repository
from .common import project_root, write_text, write_yaml
from .evidence import build_evidence_map
from .export import export_results
from .freeze import build_frozen_registry
from .integrity import build_integrity_report


def build_paper_package(root: str | Path | None = None) -> dict[str, Any]:
    project = project_root(root)
    _write_reproducibility(project)
    audit = audit_repository(project)
    evidence = build_evidence_map(project)
    frozen = build_frozen_registry(project)
    exports = export_results(project)
    _write_submission_files(project, audit, evidence, frozen)
    write_text(
        project / "docs/limitations.md",
        "# Limitations\n\n"
        "- The original interface-gap claim is falsified.\n"
        "- The supported relation-evidence finding is bounded to frozen LLaVA-1.5-7B.\n"
        "- No valid OOD, cross-model, semantic-rescue, or method claim exists.\n"
        "- Historical training, benchmark-matrix, and ablation modules are inactive, not pending.\n"
        "- The package is a negative-evidence resource, not a claim-bearing paper package.\n",
    )
    integrity = build_integrity_report(project)
    readiness = {
        "schema_version": 2,
        "as_of_experiment": "PIVOT_EXP_A3P",
        "package_status": "governance_complete_research_line_closed",
        "program_decision": "TERMINATE_CURRENT_PROGRAM",
        "scientific_submission_decision": "NO-GO",
        "scientific_submission_ready": False,
        "claim_bearing_paper_allowed": False,
        "non_claim_technical_report_allowed": True,
        "engineering_artifacts_ready": integrity["engineering_artifacts_ready"],
        "evidence_map": evidence["summary"],
        "inactive_historical_plans": {
            "recoalign_training": True,
            "comprehensive_432_cell_matrix": True,
            "mechanistic_method_evaluation": True,
        },
        "final_pivot_gate": {
            "selected_candidate": "STOP",
            "passing_candidates": [],
            "novelty_status": "VERIFIED_NO_CANDIDATE_PASSES",
        },
        "authorization": frozen["authorization"],
        "tag_created": False,
        "tag_blocker": "Claim-bearing paper and submission are not authorized.",
        "blockers": [
            "original Structured Reasoning Interface Gap is falsified",
            "no conditional causal mechanism was identified",
            "M1 and M2 semantic manipulations were falsified",
            "no candidate passed every final selection and novelty gate",
            "no valid method, cross-model replication, or OOD claim exists",
        ],
    }
    write_yaml(project / "reports/submission_readiness.yaml", readiness)
    write_text(
        project / "reports/submission_readiness_report.md",
        _readiness_markdown(readiness, integrity),
    )
    return {
        "output_root": "reports",
        "audit": audit,
        "evidence_summary": evidence["summary"],
        "frozen_registry_status": frozen["freeze_status"],
        "exports": exports,
        "integrity": integrity,
        "decision": "NO-GO",
    }


def _write_reproducibility(project: Path) -> None:
    destination = project / "reproducibility"
    destination.mkdir(parents=True, exist_ok=True)
    environment = _sanitize_environment(collect_environment(project))
    (destination / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_text(
        destination / "environment.yml",
        "\n".join(
            [
                "name: recoalign-paper-ready",
                "channels:",
                "  - conda-forge",
                "dependencies:",
                "  - python=3.10",
                "  - pip",
                "  - pip:",
                "    - -e .",
                "    - pytest>=8.0",
                "    - ruff>=0.6",
            ]
        ),
    )
    write_text(
        destination / "requirements.txt",
        "\n".join(
            [
                "numpy>=1.24",
                "Pillow>=9.0",
                "PyYAML>=6.0",
                "jsonschema>=4.21",
                "tqdm>=4.65",
                "pytest>=8.0",
                "ruff>=0.6",
            ]
        ),
    )
    write_text(
        destination / "hardware.md",
        "# Hardware and runtime\n\nSee `environment.json` for the captured Python, package, Git, and accelerator metadata. "
        "Real VLM runs require a compatible CUDA device and the optional `vlm` dependencies.",
    )
    write_text(
        destination / "dataset_protocol.md",
        "# Dataset protocol\n\nDataset versions, split rules, manifests, and leakage assertions are registered under "
        "`research/experiments/` and `manifests/datasets/`. The frozen benchmark matrix disallows "
        "test-set tuning and split overrides.",
    )
    write_text(
        destination / "training_protocol.md",
        "# Training protocol\n\nTRAIN001-TRAIN003 are inactive historical/toy contracts retained for provenance. "
        "The closed program does not authorize real-model training.",
    )
    write_text(
        destination / "evaluation_protocol.md",
        "# Evaluation protocol\n\nAll models use the BaseVLM interface, shared prompts, deterministic decoding, "
        "registered seeds, and paired statistics. Historical unexecuted cells are inactive rather "
        "than pending or imputed.",
    )
    write_text(
        destination / "README.md",
        "# Reproducibility package\n\nThis package records the environment, frozen protocols, real-VLM "
        "NO-GO evidence, and closed research-line boundary. It does not claim a successful method.",
    )


def _sanitize_environment(environment: dict[str, Any]) -> dict[str, Any]:
    """Keep relevant versions while removing user paths and unrelated global packages."""

    allowed = {
        "accelerate",
        "jsonschema",
        "numpy",
        "open-clip-torch",
        "pillow",
        "pytest",
        "pyyaml",
        "ruff",
        "sentencepiece",
        "torch",
        "torchvision",
        "tqdm",
        "transformers",
    }
    retained: list[str] = []
    for row in environment.get("pip_freeze", []):
        normalized = str(row).split("==", 1)[0].split("[", 1)[0].lower().replace("_", "-")
        if (
            normalized in allowed
            and "file:" not in str(row).lower()
            and "git+" not in str(row).lower()
        ):
            retained.append(str(row))
    environment["pip_freeze"] = retained
    environment["pip_freeze_sha256"] = hashlib.sha256(
        "\n".join(retained).encode("utf-8")
    ).hexdigest()
    executable = environment.get("executable")
    environment["executable"] = Path(str(executable)).name if executable else None
    prefix = environment.get("conda_prefix")
    environment["conda_prefix"] = Path(str(prefix)).name if prefix else None
    environment["privacy_filter"] = {
        "local_paths_removed": True,
        "unrelated_global_packages_removed": True,
        "author_identity_not_recorded": True,
    }
    return environment


def _write_submission_files(
    project: Path, audit: dict[str, Any], evidence: dict[str, Any], frozen: dict[str, Any]
) -> None:
    destination = project / "submission"
    destination.mkdir(parents=True, exist_ok=True)
    write_text(
        destination / "code_structure.md",
        "# Anonymous code structure\n\n"
        "- `src/recoalign/`: governed experiment, model, training, analysis, and packaging code\n"
        "- `configs/`: versioned protocol and model configuration\n"
        "- `research/`: hypothesis, experiment, and training registries\n"
        "- `manifests/`: dataset and checkpoint identities\n"
        "- `reports/`: generated evidence and readiness artifacts\n\n"
        "The method line is closed; release is authorized only as a negative-evidence resource.",
    )
    write_text(
        destination / "reproducibility_checklist.md",
        "# Reproducibility checklist\n\n"
        "- [x] Configuration and manifest identities are recorded\n"
        "- [x] Prompt and decoding protocol is versioned\n"
        "- [x] Environment capture is generated\n"
        "- [x] Missing/failed evidence remains visible\n"
        "- [x] PIVOT_EXP_A through A3P failures retained\n"
        "- [x] Cross-backbone replication marked unauthorized\n"
        "- [x] Comprehensive matrix marked inactive\n"
        "- [x] ReCoAlign training and method claims retired\n"
        "- [x] Claim-bearing paper writing blocked\n",
    )
    write_text(
        destination / "artifact_description.md",
        "# Artifact description\n\n"
        "The package contains executable governance, frozen negative evidence, synthetic benchmark "
        "provenance, and conservative claim mapping. Historical training and ablation assets are "
        "inactive.\n\n"
        f"Claims mapped: {evidence['summary']['total_claims']}; verified: {evidence['summary']['verified']}.\n"
        f"Frozen registry status: {frozen['freeze_status']}.\n",
    )
    write_text(
        destination / "limitations.md",
        "# Limitations\n\n"
        "- The interface-gap claim is falsified.\n"
        "- The positive relation-evidence result is limited to frozen LLaVA-1.5-7B.\n"
        "- No valid OOD, cross-model, semantic-rescue, or method claim exists.\n"
        "- Historical adapters and matrices are inactive and not authorized for execution.\n",
    )
    write_text(
        destination / "README.md",
        "# ReCoAlign anonymous research package\n\n"
        "This is a closed-line negative-evidence artifact, not a paper-ready method package. See "
        "`reports/final_research_line_adjudication.md`, `reports/submission_readiness_report.md`, "
        "and `docs/evidence_map.yaml`.",
    )
    del audit


def _readiness_markdown(readiness: dict[str, Any], integrity: dict[str, Any]) -> str:
    blockers = "\n".join(f"- {blocker}" for blocker in readiness["blockers"])
    return "\n".join(
        [
            "# Submission readiness report",
            "",
            f"- Package implementation: **{readiness['package_status']}**",
            f"- Engineering artifact status: **{'GO' if readiness['engineering_artifacts_ready'] else 'INCONCLUSIVE'}**",
            f"- Scientific submission decision: **{readiness['scientific_submission_decision']}**",
            f"- Verified claims: **{readiness['evidence_map']['verified']}/{readiness['evidence_map']['total_claims']}**",
            "- Historical comprehensive matrix: **inactive**",
            "",
            "## Scientific contribution",
            "",
            "The frozen evidence rejects the original Structured Reasoning Interface Gap. The "
            "research line is closed and supports no method or efficacy claim.",
            "",
            "## Method summary",
            "",
            "No active method is claimed. No new experiment or preregistration is authorized.",
            "",
            "## Experimental evidence",
            "",
            "Frozen evidence is present through PIVOT_EXP_A3P. EXP001/EXP004 are NO-GO, EXP002 is "
            "scope-bounded GO, EXP003/A2 are inconclusive, A3/A3R instruments are retired, and A3P "
            "reports semantic manipulation failure.",
            "",
            "## Reproducibility status",
            "",
            f"- Integrity report: `{integrity['path']}`",
            "- Environment, protocol, manifest, and submission checklists generated.",
            "- Release tag not created because the scientific gate is not satisfied.",
            "",
            "## Remaining risks",
            "",
            blockers,
            "",
            "## Decision",
            "",
            "**TERMINATE_CURRENT_PROGRAM; NO-GO for claim-bearing paper writing or submission.**",
        ]
    )
