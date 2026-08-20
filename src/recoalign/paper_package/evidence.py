# ruff: noqa: E501
"""Claim-to-artifact mapping with conservative scientific eligibility rules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import load_json, load_yaml, project_root, write_yaml

CLAIMS: tuple[dict[str, Any], ...] = (
    {
        "id": "C001",
        "claim": "A structured graph interface improves reasoning over information-controlled text.",
        "evidence": ["paired graph-versus-caption evaluation", "multi-seed uncertainty"],
        "experiments": ["EXP001"],
        "artifacts": [
            "configs/graph_vs_text.yaml",
            "research/protocols/EXP001_graph_vs_text.md",
            "reports/experiments/exp001/implementation_report.md",
        ],
        "status": "infrastructure_only",
        "blocker": "No claim-eligible real-VLM multi-seed EXP001 result is complete.",
    },
    {
        "id": "C002",
        "claim": "Reasoning gains depend on correct relational structure rather than graph-like form.",
        "evidence": ["partial/corrupt/random graph controls", "token-length matching"],
        "experiments": ["EXP002"],
        "artifacts": [
            "configs/graph_ablation.yaml",
            "research/protocols/EXP002_graph_ablation.md",
        ],
        "status": "infrastructure_only",
        "blocker": "EXP002 results use an infrastructure ReferenceVLM, not an eligible real VLM.",
    },
    {
        "id": "C003",
        "claim": "Structured reasoning improves compositional OOD retention and depth transfer.",
        "evidence": ["composition-disjoint split", "OOD retention", "hop-depth curve"],
        "experiments": ["EXP003"],
        "artifacts": [
            "configs/ood_composition.yaml",
            "research/protocols/EXP003_ood_composition.md",
            "docs/composition_split_protocol.md",
        ],
        "status": "infrastructure_only",
        "blocker": "No claim-eligible real-VLM EXP003 matrix has been completed.",
    },
    {
        "id": "C004",
        "claim": "Real VLMs preserve semantics but exhibit a structured reasoning interface gap.",
        "evidence": ["SAS/StAS/RES diagnosis", "oracle-structure gain across VLMs"],
        "experiments": ["EXP004"],
        "artifacts": [
            "configs/diagnosis/interface_diagnosis.yaml",
            "reports/diagnosis/mechanism_diagnosis.md",
        ],
        "status": "pending",
        "blocker": "EXP004 is a ReferenceVLM infrastructure run; real hidden-state access is pending.",
    },
    {
        "id": "C005",
        "claim": "ReCoAlign learns a graph-free structured interface from training supervision.",
        "evidence": ["TRAIN001-TRAIN003", "graph-free inference contract", "checkpoint audit"],
        "experiments": ["TRAIN001", "TRAIN002", "TRAIN003"],
        "artifacts": [
            "research/experiments/training_registry.yaml",
            "reports/training_framework/training_framework_report.md",
            "reports/training_framework/decision_report.yaml",
        ],
        "status": "infrastructure_only",
        "blocker": "Toy optimization passes, but real-VLM interface training remains pending.",
    },
    {
        "id": "C006",
        "claim": "ReCoAlign improves compositional reasoning across models and tasks without severe capability loss.",
        "evidence": ["frozen 432-cell matrix", "paired statistics", "capability preservation"],
        "experiments": ["PHASE4_COMPREHENSIVE"],
        "artifacts": [
            "configs/benchmarks/comprehensive_matrix.yaml",
            "reports/comprehensive_evaluation_report.md",
            "reports/decision_report.json",
        ],
        "status": "pending",
        "blocker": "The comprehensive matrix currently has 0 of 432 complete cells.",
    },
    {
        "id": "C007",
        "claim": "The gain is causally attributable to learned structure, not parameters or oracle input.",
        "evidence": ["registered ablations", "token interventions", "parameter-matched control"],
        "experiments": ["PHASE5_MECHANISTIC"],
        "artifacts": [
            "configs/ablations/mechanistic_registry.yaml",
            "reports/mechanistic_validation_report.md",
            "reports/mechanistic/decision_report.yaml",
        ],
        "status": "pending",
        "blocker": "Only toy causal evidence is available; real multi-seed interventions are pending.",
    },
)


def build_evidence_map(root: str | Path | None = None) -> dict[str, Any]:
    project = project_root(root)
    claims = [dict(claim) for claim in CLAIMS]
    for claim in claims:
        claim["artifact_checks"] = [
            {"path": artifact, "exists": (project / artifact).is_file()}
            for artifact in claim["artifacts"]
        ]
        claim["artifact_inventory_complete"] = all(
            row["exists"] for row in claim["artifact_checks"]
        )

    comprehensive = project / "reports/decision_report.json"
    if comprehensive.is_file():
        decision = load_json(comprehensive)
        for claim in claims:
            if claim["id"] == "C006":
                claim["observed_cells"] = decision.get("complete_cells", 0)
                claim["planned_cells"] = decision.get("planned_cells", 432)

    mechanism = project / "reports/mechanistic/decision_report.yaml"
    if mechanism.is_file():
        decision = load_yaml(mechanism)
        for claim in claims:
            if claim["id"] == "C007":
                claim["current_decision"] = decision.get("decision")
                claim["real_vlm_evidence"] = decision.get("real_vlm_mechanistic_evidence")

    payload = {
        "schema_version": 1,
        "policy": {
            "verified_requires_scientific_evidence": True,
            "toy_or_reference_results_are_not_claim_eligible": True,
            "failed_and_pending_evidence_remains_visible": True,
        },
        "summary": {
            "total_claims": len(claims),
            "verified": sum(claim["status"] == "verified" for claim in claims),
            "infrastructure_only": sum(
                claim["status"] == "infrastructure_only" for claim in claims
            ),
            "pending": sum(claim["status"] == "pending" for claim in claims),
        },
        "claims": claims,
    }
    write_yaml(project / "docs/evidence_map.yaml", payload)
    return payload
