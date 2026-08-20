"""Phase-6 paper package integrity and conservative decision tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from recoalign.paper_package import FreezeError, build_paper_package, create_paper_tag
from recoalign.paper_package.evidence import build_evidence_map
from recoalign.paper_package.export import export_results
from recoalign.paper_package.freeze import build_frozen_registry
from recoalign.paper_package.integrity import build_integrity_report

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def preserve_committed_frozen_registry() -> Iterator[None]:
    """Keep package-generation tests from dirtying the checked-out repository."""

    paths = (
        ROOT / "experiments/frozen_registry.yaml",
        ROOT / "reports/final_repository_audit.md",
    )
    originals = {
        path: path.read_bytes() if path.is_file() else None
        for path in paths
    }
    try:
        yield
    finally:
        for path, original in originals.items():
            if original is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(original)


def test_evidence_map_reflects_frozen_real_vlm_outcomes() -> None:
    payload = build_evidence_map(ROOT)
    assert payload["summary"] == {
        "total_claims": 7,
        "verified": 1,
        "falsified": 2,
        "inconclusive": 1,
        "infrastructure_only": 1,
        "pending": 2,
    }
    statuses = {claim["id"]: claim["status"] for claim in payload["claims"]}
    assert statuses["C001"] == "falsified"
    assert statuses["C002"] == "verified"
    assert statuses["C003"] == "inconclusive"
    assert statuses["C004"] == "falsified"


def test_frozen_registry_records_real_evidence_and_pending_training() -> None:
    payload = build_frozen_registry(ROOT)
    assert payload["freeze_status"] == "evidence_frozen_no_go"
    assert payload["protocol_changes_allowed"] is False
    assert {row["experiment_id"] for row in payload["experiments"]} == {
        "EXP001",
        "EXP002",
        "EXP003",
        "EXP004",
        "TRAIN001",
        "TRAIN002",
        "TRAIN003",
    }
    assert all(row["config_sha256"] for row in payload["experiments"])
    evaluation = [row for row in payload["experiments"] if row["kind"] == "evaluation"]
    training = [row for row in payload["experiments"] if row["kind"] == "training"]
    assert all(row["checkpoint_sha256"] for row in evaluation)
    assert all(row["checkpoint_fingerprint"] for row in evaluation)
    assert all(row["checkpoint_sha256"] is None for row in training)


def test_exports_mark_missing_results_without_placeholder_numbers() -> None:
    payload = export_results(ROOT)
    assert payload["complete_matrix_cells"] == 0
    assert payload["planned_matrix_cells"] == 432
    assert payload["contains_placeholder_numbers"] is False
    text = (ROOT / "reports/results/main_results.tex").read_text(encoding="utf-8")
    assert "Evidence pending" in text


def test_integrity_is_engineering_ready_but_scientific_no_go() -> None:
    build_paper_package(ROOT)
    payload = build_integrity_report(ROOT)
    assert payload["engineering_artifacts_ready"] is True
    assert payload["scientific_submission_ready"] is False
    assert payload["decision"] == "NO-GO"


def test_tag_creation_is_blocked_until_scientific_go() -> None:
    build_paper_package(ROOT)
    with pytest.raises(FreezeError, match="decision is not GO"):
        create_paper_tag(ROOT)


def test_anonymous_environment_has_no_user_path() -> None:
    build_paper_package(ROOT)
    text = (ROOT / "reproducibility/environment.json").read_text(encoding="utf-8").lower()
    assert "c:\\users" not in text
    assert "yangjunjie" not in text
    readiness = yaml.safe_load(
        (ROOT / "reports/submission_readiness.yaml").read_text(encoding="utf-8")
    )
    assert readiness["scientific_submission_decision"] == "NO-GO"
    assert readiness["tag_created"] is False
