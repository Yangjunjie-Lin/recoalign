"""Regression checks for the frozen research identity and narrative boundary."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_identity_artifacts_exist_and_agree() -> None:
    identity = (ROOT / "docs/research_identity.md").read_text(encoding="utf-8")
    hypothesis = yaml.safe_load(
        (ROOT / "research/frozen_hypothesis.yaml").read_text(encoding="utf-8")
    )
    assert "Structured Reasoning Interface Learning" in identity
    assert hypothesis["status"] == "frozen"
    assert hypothesis["project"] == "ReCoAlign"
    assert "structured intermediate representation" in hypothesis["main_hypothesis"]
    assert {row["id"] for row in hypothesis["supporting_claims"]} == {"C1", "C2", "C3", "C4"}


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
    assert "structured reasoning interface" in text
    assert "typed hard negatives" not in text
    assert "region–phrase alignment" not in text
    assert "retrieval-first" not in text


def test_readme_marks_retrieval_as_a_control() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "capability-preservation" in readme
    assert "not the new scientific contribution" in readme
    assert "docs/research_identity.md" in readme
