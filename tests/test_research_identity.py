"""Regression checks for the rejected freeze and final research-line closeout."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_rejected_freeze_and_pivot_identity_artifacts_agree() -> None:
    identity = (ROOT / "docs/research_identity.md").read_text(encoding="utf-8")
    hypothesis = yaml.safe_load(
        (ROOT / "research/frozen_hypothesis.yaml").read_text(encoding="utf-8")
    )
    pivot = yaml.safe_load((ROOT / "research/pivot_hypothesis.yaml").read_text(encoding="utf-8"))
    closeout = yaml.safe_load(
        (ROOT / "research/current_line_closeout.yaml").read_text(encoding="utf-8")
    )
    assert "closed negative-evidence" in identity
    assert hypothesis["status"] == "falsified"
    assert hypothesis["project"] == "ReCoAlign"
    assert "structured intermediate representation" in hypothesis["main_hypothesis"]
    assert {row["id"] for row in hypothesis["supporting_claims"]} == {"C1", "C2", "C3", "C4"}
    assert pivot["status"] == "unresolved_and_not_identified"
    assert pivot["lifecycle"] == "retired"
    assert pivot["paper_writing_allowed"] is False
    assert closeout["program_decision"] == "TERMINATE_CURRENT_PROGRAM"
    assert closeout["final_pivot_gate"]["selected_candidate"] == "STOP"


def test_active_research_plan_does_not_reintroduce_old_method_identity() -> None:
    active_docs = (
        ROOT / "README.md",
        ROOT / "docs/research_plan.md",
        ROOT / "docs/research_identity.md",
        ROOT / "docs/contribution_framework.md",
        ROOT / "docs/claim_boundary.md",
        ROOT / "docs/architecture.md",
        ROOT / "docs/experiment_protocol.md",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in active_docs).lower()
    boundary = (ROOT / "docs/claim_boundary.md").read_text(encoding="utf-8").lower()
    assert "semantic" in text and "integration" in text
    assert "rejected" in text or "falsified" in text
    assert "prohibited claims" in boundary
    assert "graph is the missing" in boundary and "optimal" in boundary
    assert "typed hard negatives" not in text
    assert "region–phrase alignment" not in text
    assert "retrieval-first" not in text


def test_readme_marks_retrieval_as_a_control() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "capability-preservation" in readme
    assert "not the new scientific contribution" in readme
    assert "docs/research_identity.md" in readme


def test_final_candidate_registry_selects_stop_only() -> None:
    registry = yaml.safe_load(
        (ROOT / "research/hypotheses/pivot_candidate_registry.yaml").read_text(encoding="utf-8")
    )
    matrix = yaml.safe_load(
        (ROOT / "research/final_pivot_candidate_matrix.yaml").read_text(encoding="utf-8")
    )
    assert registry["selected_candidate"] == "STOP"
    assert registry["passing_candidates"] == []
    assert matrix["selected_candidate"] == "STOP"
    assert not any(
        row["passes_all_selection_gates"] is True
        for row in matrix["candidates"]
        if row["id"] != "STOP"
    )


def test_frozen_hypothesis_lifecycle_matches_real_decisions() -> None:
    registry = yaml.safe_load(
        (ROOT / "research/hypotheses/hypothesis_registry.yaml").read_text(encoding="utf-8")
    )
    statuses = {row["id"]: row["status"] for row in registry["hypotheses"]}
    assert statuses == {
        "H001": "falsified",
        "H002": "supported_in_frozen_LLaVA_scope_only",
        "H003": "retired",
        "H004": "falsified",
    }
