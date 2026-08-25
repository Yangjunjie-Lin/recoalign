# ruff: noqa: E501
"""Closed-line claim-to-artifact map through PIVOT_EXP_A3P."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import load_yaml, project_root, write_yaml

CLAIMS: tuple[dict[str, Any], ...] = (
    {
        "id": "C001",
        "claim": "A structured graph interface improves reasoning over information-controlled text.",
        "experiments": ["EXP001"],
        "artifacts": ["reports/paper_evidence/EXP001/metrics.json", "reports/paper_evidence/EXP001/decision_report.yaml"],
        "status": "falsified",
        "registered_decision": "NO-GO",
        "scope": "frozen LLaVA-1.5-7B execution",
        "interpretation": "Caption outperformed Graph; no privileged graph interface was established.",
    },
    {
        "id": "C002",
        "claim": "Correct supplied relational evidence affects reasoning more than corrupted, partial, or random relational evidence.",
        "experiments": ["EXP002"],
        "artifacts": ["reports/paper_evidence/EXP002/metrics.json", "reports/paper_evidence/EXP002/decision_report.yaml"],
        "status": "verified",
        "registered_decision": "GO",
        "scope": "frozen LLaVA-1.5-7B execution only",
        "interpretation": "This is external evidence sensitivity, not proof of an internal scene graph or a general VLM mechanism.",
    },
    {
        "id": "C003",
        "claim": "Structured evidence improves compositional OOD retention and depth transfer.",
        "experiments": ["EXP003"],
        "artifacts": ["reports/paper_evidence/EXP003/metrics.json", "reports/paper_evidence/EXP003/decision_report.yaml"],
        "status": "inconclusive",
        "registered_decision": "INCONCLUSIVE",
        "scope": "no OOD claim eligible",
        "interpretation": "The preregistered token-length integrity gate failed.",
    },
    {
        "id": "C004",
        "claim": "Visual semantics are highly available while structured accessibility is deficient.",
        "experiments": ["EXP004"],
        "artifacts": ["reports/paper_evidence/EXP004/metrics.json", "reports/paper_evidence/EXP004/decision_report.yaml"],
        "status": "falsified",
        "registered_decision": "NO-GO",
        "scope": "registered probe and frozen LLaVA-1.5-7B execution",
        "interpretation": "Low SAS rejects the high-availability premise; it does not prove semantic absence at every hidden layer.",
    },
    {
        "id": "C005",
        "claim": "ReCoAlign learns a graph-free structured interface from training supervision.",
        "experiments": ["TRAIN001", "TRAIN002", "TRAIN003"],
        "artifacts": ["research/experiments/training_registry.yaml"],
        "status": "retired",
        "registered_decision": "NOT_AUTHORIZED",
        "scope": "historical toy/infrastructure assets only",
        "interpretation": "No claim-eligible trained method exists; real training is inactive, not pending.",
    },
    {
        "id": "C006",
        "claim": "ReCoAlign improves compositional reasoning across models and tasks without severe capability loss.",
        "experiments": ["PHASE4_COMPREHENSIVE"],
        "artifacts": ["configs/benchmarks/comprehensive_matrix.yaml"],
        "status": "retired",
        "registered_decision": "NOT_AUTHORIZED",
        "scope": "unexecuted historical plan",
        "interpretation": "The 432-cell matrix is inactive and is not an authorized next step.",
    },
    {
        "id": "C007",
        "claim": "A ReCoAlign gain is causally attributable to a learned structured interface.",
        "experiments": ["PHASE5_MECHANISTIC"],
        "artifacts": ["configs/ablations/mechanistic_registry.yaml"],
        "status": "retired",
        "registered_decision": "NOT_AUTHORIZED",
        "scope": "unexecuted method claim",
        "interpretation": "Mechanistic method evaluation is inactive because method development lacks scientific authorization.",
    },
    {
        "id": "C008",
        "claim": "The joint semantic-structural phenotype identifies a dominant causal mechanism.",
        "experiments": ["PIVOT_EXP_A", "PIVOT_EXP_A2"],
        "artifacts": ["research/pivot_validation/results/pivot_decision.yaml", "research/causal_separation/PIVOT_EXP_A2/results/decision_report.yaml"],
        "status": "falsified",
        "registered_decision": "NO-GO / INCONCLUSIVE",
        "scope": "causal identification",
        "interpretation": "PIVOT_EXP_A supported multiple mechanisms, and A2 failed semantic-sufficiency and parsing-integrity gates.",
    },
    {
        "id": "C009",
        "claim": "The original full-field free-generation answer contract is valid.",
        "experiments": ["PIVOT_EXP_A3"],
        "experiment": "PIVOT_EXP_A3",
        "artifacts": ["research/construct_validity/PIVOT_EXP_A3/blocked_preinference_record.yaml"],
        "status": "falsified_as_measurement_instrument",
        "registered_decision": "RUNTIME_BLOCKED_PREINFERENCE",
        "scope": "measurement instrument only",
        "interpretation": "The frozen full-field contract failed in development smoke before validation inference.",
    },
    {
        "id": "C010",
        "claim": "The continuation-form free-generation contract is valid.",
        "experiments": ["PIVOT_EXP_A3R"],
        "experiment": "PIVOT_EXP_A3R",
        "artifacts": ["research/construct_validity/PIVOT_EXP_A3R/final_instrument_retirement_record.yaml"],
        "status": "falsified_as_measurement_instrument",
        "registered_decision": "SECONDARY_CONTRACT_RUNTIME_FAILURE",
        "scope": "measurement instrument only",
        "interpretation": "127/160 development responses were valid; 33 entity-ID continuations caused retirement before validation.",
    },
    {
        "id": "C011",
        "claim": "The frozen primary conditional-likelihood measurement is technically valid.",
        "experiments": ["PIVOT_EXP_A3P"],
        "experiment": "PIVOT_EXP_A3P",
        "artifacts": ["research/construct_validity/PIVOT_EXP_A3P/results/primary_measurement_report.yaml", "research/construct_validity/PIVOT_EXP_A3P/results/artifact_manifest.yaml", "research/construct_validity/PIVOT_EXP_A3P/results/predictions.jsonl.gz"],
        "status": "verified",
        "registered_decision": "MEASUREMENT_INTEGRITY_PASS",
        "scope": "measurement integrity only",
        "interpretation": "The scorer and 27,000-row prediction inventory are valid; semantic comprehension is not verified.",
    },
    {
        "id": "C012",
        "claim": "M1 canonical entity table establishes sufficient semantic rescue.",
        "experiments": ["PIVOT_EXP_A3P"],
        "experiment": "PIVOT_EXP_A3P",
        "artifacts": ["research/construct_validity/PIVOT_EXP_A3P/results/decision_report.yaml"],
        "status": "falsified",
        "registered_decision": "SEMANTIC_MANIPULATION_FAILURE",
        "scope": "frozen M1 manipulation",
        "interpretation": "M1 failed Gates C, D, and E; partial positive effects do not constitute semantic rescue.",
    },
    {
        "id": "C013",
        "claim": "M2 visual object legend establishes sufficient semantic rescue.",
        "experiments": ["PIVOT_EXP_A3P"],
        "experiment": "PIVOT_EXP_A3P",
        "artifacts": ["research/construct_validity/PIVOT_EXP_A3P/results/decision_report.yaml"],
        "status": "falsified",
        "registered_decision": "SEMANTIC_MANIPULATION_FAILURE",
        "scope": "frozen M2 manipulation",
        "interpretation": "M2 failed Gates C, D, and E; passing Gate F cannot rescue the manipulation.",
    },
    {
        "id": "C014",
        "claim": "A valid semantic manipulation has been identified for causal mechanism separation.",
        "experiments": ["PIVOT_EXP_A3P"],
        "experiment": "PIVOT_EXP_A3P",
        "artifacts": ["research/construct_validity/PIVOT_EXP_A3P/results/decision_report.yaml"],
        "status": "falsified",
        "registered_decision": "SEMANTIC_MANIPULATION_FAILURE",
        "scope": "construct selection",
        "interpretation": "Neither M1 nor M2 passed all upstream semantic gates; no manipulation was selected.",
    },
)


def build_evidence_map(root: str | Path | None = None) -> dict[str, Any]:
    project = project_root(root)
    claims = [dict(claim) for claim in CLAIMS]
    _verify_original_frozen_outcomes(project, claims)
    for claim in claims:
        claim["evidence_role"] = "scientific_evidence" if claim["status"] != "retired" else "provenance_only"
        claim["artifact_checks"] = [
            {"path": artifact, "exists": (project / artifact).is_file()}
            for artifact in claim["artifacts"]
        ]
        claim["artifact_inventory_complete"] = all(row["exists"] for row in claim["artifact_checks"])

    status_counts = {
        "verified": sum(c["status"] == "verified" for c in claims),
        "falsified": sum(c["status"] == "falsified" for c in claims),
        "falsified_as_measurement_instrument": sum(c["status"] == "falsified_as_measurement_instrument" for c in claims),
        "inconclusive": sum(c["status"] == "inconclusive" for c in claims),
        "retired": sum(c["status"] == "retired" for c in claims),
    }
    payload = {
        "schema_version": 2,
        "as_of_experiment": "PIVOT_EXP_A3P",
        "program_decision": "TERMINATE_CURRENT_PROGRAM",
        "policy": {
            "measurement_validity_is_not_semantic_validity": True,
            "single_backbone_findings_remain_scope_bounded": True,
            "failed_and_inconclusive_evidence_remains_visible": True,
            "inactive_method_work_is_not_pending": True,
        },
        "summary": {"total_claims": len(claims), **status_counts, "pending": 0},
        "claims": claims,
    }
    write_yaml(project / "docs/evidence_map.yaml", payload)
    return payload


def _verify_original_frozen_outcomes(project: Path, claims: list[dict[str, Any]]) -> None:
    """Cross-check C001-C004 against the immutable paper-evidence manifest."""
    manifest_path = project / "reports/paper_evidence/artifact_manifest.yaml"
    if not manifest_path.is_file():
        return
    experiments = load_yaml(manifest_path).get("experiments", {})
    by_claim = {"C001": "EXP001", "C002": "EXP002", "C003": "EXP003", "C004": "EXP004"}
    for claim in claims:
        experiment_id = by_claim.get(claim["id"])
        if experiment_id is None or experiment_id not in experiments:
            continue
        result = experiments[experiment_id]
        claim["frozen_manifest_status"] = result.get("status")
        claim["frozen_manifest_decision"] = result.get("decision")
