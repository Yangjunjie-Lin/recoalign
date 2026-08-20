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
        "- The frozen LLaVA-1.5-7B evidence does not support the main interface-gap claim.\n"
        "- Stage-1 structural labels come from a synthetic world; real-task transfer remains unverified.\n"
        "- LLaVA-NeXT, Qwen-VL, and InternVL are adapter-ready but lack complete claim-eligible runs.\n"
        "- The comprehensive benchmark and mechanistic matrices are incomplete.\n"
        "- ReCoAlign itself has no claim-eligible trained real-VLM checkpoint.\n",
    )
    integrity = build_integrity_report(project)
    decision = "GO" if integrity["scientific_submission_ready"] else "NO-GO"
    evidence_check = integrity["checks"].get("claim_evidence", {})
    evidence_decision = str(evidence_check.get("scientific_decision", "pending")).lower()
    real_vlm_evidence = (
        f"completed_{evidence_decision.replace('-', '_')}"
        if evidence_check.get("passed") is True
        else "pending_or_invalid"
    )
    readiness = {
        "schema_version": 1,
        "package_status": "implementation_complete",
        "scientific_submission_decision": decision,
        "scientific_submission_ready": integrity["scientific_submission_ready"],
        "engineering_artifacts_ready": integrity["engineering_artifacts_ready"],
        "claims_verified": evidence["summary"]["verified"],
        "claims_total": evidence["summary"]["total_claims"],
        "complete_matrix_cells": exports["complete_matrix_cells"],
        "planned_matrix_cells": exports["planned_matrix_cells"],
        "real_vlm_evidence": real_vlm_evidence,
        "tag_created": False,
        "tag_blocker": "Tag creation requires GO, clean worktree, and a committed frozen snapshot.",
        "blockers": integrity["blockers"],
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
        "decision": decision,
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
        "# Training protocol\n\nTRAIN001-TRAIN003 define the staged training contract. Graph supervision is a "
        "training signal only; ReCoAlign inference must remain graph-free.",
    )
    write_text(
        destination / "evaluation_protocol.md",
        "# Evaluation protocol\n\nAll models use the BaseVLM interface, shared prompts, deterministic decoding, "
        "registered seeds, and paired statistics. Missing cells are reported as pending rather than "
        "imputed.",
    )
    write_text(
        destination / "README.md",
        "# Reproducibility package\n\nThis package records the environment and frozen protocols. It does not "
        "claim that pending real-VLM experiments have been completed.",
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
        "The current worktree is a candidate package; the release tag is intentionally not created while the scientific decision is NO-GO.",
    )
    write_text(
        destination / "reproducibility_checklist.md",
        "# Reproducibility checklist\n\n"
        "- [x] Configuration and manifest identities are recorded\n"
        "- [x] Prompt and decoding protocol is versioned\n"
        "- [x] Environment capture is generated\n"
        "- [x] Missing/failed evidence remains visible\n"
        "- [ ] Complete real-VLM multi-seed benchmark matrix\n"
        "- [ ] Claim-eligible ReCoAlign checkpoints for all target backbones\n"
        "- [ ] Clean committed release snapshot\n",
    )
    write_text(
        destination / "artifact_description.md",
        "# Artifact description\n\n"
        "The package contains executable governance, synthetic benchmark generation, VLM adapters, "
        "training/ablation controls, reporting utilities, and conservative claim mapping. Toy and "
        "ReferenceVLM outputs are labeled infrastructure evidence only.\n\n"
        f"Claims mapped: {evidence['summary']['total_claims']}; verified: {evidence['summary']['verified']}.\n"
        f"Frozen registry status: {frozen['freeze_status']}.\n",
    )
    write_text(
        destination / "limitations.md",
        "# Limitations\n\n"
        "- Real VLM inference is compute- and memory-intensive.\n"
        "- The LLaVA diagnosis accesses visual hidden representations but does not support the registered interface-gap pattern.\n"
        "- Stage-1 structural supervision is synthetic; transfer must be measured rather than assumed.\n"
        "- Backbone adapters for LLaVA-NeXT, Qwen-VL, and InternVL are interface boundaries until executed.\n"
        "- The complete 432-cell benchmark matrix is not yet populated.\n",
    )
    write_text(
        destination / "README.md",
        "# ReCoAlign anonymous research package\n\n"
        "This is a submission-preparation artifact, not a claim that the pending experiments are complete. "
        "See `reports/submission_readiness_report.md` and `docs/evidence_map.yaml`.",
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
            f"- Verified claims: **{readiness['claims_verified']}/{readiness['claims_total']}**",
            f"- Comprehensive cells: **{readiness['complete_matrix_cells']}/{readiness['planned_matrix_cells']}**",
            "",
            "## Scientific contribution",
            "",
            "The repository implements a diagnosis-driven structured interface research program. "
            "It does not yet support the final efficacy claim because the required real-VLM evidence is pending.",
            "",
            "## Method summary",
            "",
            "ReCoAlign exposes a learned visual-to-structure interface and keeps oracle graph information out of inference.",
            "",
            "## Experimental evidence",
            "",
        "Frozen LLaVA-1.5-7B evidence is present: EXP001 and EXP004 are NO-GO, EXP002 is GO, and EXP003 is "
        "INCONCLUSIVE after a preregistered integrity failure. ReCoAlign training and the comprehensive matrix remain incomplete.",
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
            "**NO-GO for scientific submission; GO for paper-package implementation.**",
        ]
    )
