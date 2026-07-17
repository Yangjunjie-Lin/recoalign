import json
import shutil

import pytest
from test_experiment_records import (
    _complete_winoground_run,
    _complete_winoground_verification_run,
    _write_synthetic_prediction_review,
)

from recoalign.analysis.results import collect_runs, render_markdown_table
from recoalign.experiments.records import create_run, finalize_run, promote_run
from recoalign.reproducibility import atomic_write_json


def _make_reportable(run_dir) -> None:
    environment_path = run_dir / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["git"].update(
        {
            "commit": "a" * 40,
            "branch": "test",
            "dirty": False,
            "untracked_count": 0,
            "diff_sha256": None,
        }
    )
    atomic_write_json(environment_path, environment)

    run_path = run_dir / "run.json"
    record = json.loads(run_path.read_text(encoding="utf-8"))
    record["git_commit"] = "a" * 40
    atomic_write_json(run_path, record)
    promote_run(run_dir, reviewed_by="Reviewer")


def test_only_reportable_runs_enter_tables(tmp_path, research_config) -> None:
    reportable = create_run(research_config, output_root=tmp_path, run_id="reportable")
    finalize_run(reportable, {"R@1": 60}, status="complete")
    _make_reportable(reportable)

    pilot = create_run(research_config, output_root=tmp_path, run_id="pilot")
    finalize_run(pilot, {"R@1": 99}, status="pilot")

    records = collect_runs(tmp_path)
    table = render_markdown_table(records, ["R@1"])
    assert "reportable" in table
    assert "pilot" not in table
    assert "60.0000" in table


def test_collect_runs_revalidates_complete_winoground_promotion(
    tmp_path, research_config
) -> None:
    canonical, verification = _promoted_winoground_pair(tmp_path, research_config)

    records = collect_runs(tmp_path)
    table = render_markdown_table(records, ["group_accuracy"])

    assert canonical.name in table
    assert verification.name not in table


def test_collect_runs_rejects_hand_edited_winoground_reportable_status(
    tmp_path, research_config
) -> None:
    canonical = _complete_winoground_run(tmp_path, research_config)
    path = canonical / "run.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    run["status"] = "reportable"
    run["review"] = {
        "reviewed_by": "manual",
        "reviewed_at": "2026-07-16T00:00:00+00:00",
        "decision": "reportable",
        "notes": None,
    }
    atomic_write_json(path, run)

    with pytest.raises(ValueError):
        collect_runs(tmp_path)


@pytest.mark.parametrize(
    ("target", "mutation", "error"),
    [
        ("evidence", "missing", "promotion_evidence evidence file is missing"),
        ("comparison", "passed", "promotion evidence field mismatch: passed"),
        ("review", "mapping", "mapping_checked=true"),
        ("run", "review_hash", "comparison SHA mismatch"),
        ("evidence", "canonical_id", "canonical_run_id"),
        ("evidence", "verification_id", "verification_run_id"),
    ],
)
def test_collect_runs_rejects_tampered_promotion_evidence(
    tmp_path,
    research_config,
    target: str,
    mutation: str,
    error: str,
) -> None:
    canonical, _verification = _promoted_winoground_pair(tmp_path, research_config)
    if target == "evidence":
        path = canonical / "review" / "promotion_evidence.json"
        if mutation == "missing":
            path.unlink()
        else:
            evidence = json.loads(path.read_text(encoding="utf-8"))
            field = "canonical_run_id" if mutation == "canonical_id" else "verification_run_id"
            evidence[field] = "other"
            atomic_write_json(path, evidence)
            _update_review_digest(canonical, "promotion_evidence", path)
    elif target == "comparison":
        path = canonical / "review" / "run_comparison.json"
        comparison = json.loads(path.read_text(encoding="utf-8"))
        comparison["passed"] = False
        atomic_write_json(path, comparison)
        _synchronize_evidence_hash(canonical, "comparison", path)
    elif target == "review":
        path = canonical / "review" / "prediction_review.csv"
        path.write_text(
            path.read_text(encoding="utf-8").replace(",true,", ",false,", 1),
            encoding="utf-8",
        )
        _synchronize_evidence_hash(canonical, "prediction_review", path)
    else:
        path = canonical / "run.json"
        run = json.loads(path.read_text(encoding="utf-8"))
        run["review"]["comparison_sha256"] = "b" * 64
        atomic_write_json(path, run)

    with pytest.raises(ValueError, match=error):
        collect_runs(tmp_path)


@pytest.mark.parametrize(
    ("action", "error"),
    [
        ("missing", "verification run is missing"),
        ("duplicate", "multiple verification runs"),
        ("reportable", "validation failed"),
    ],
)
def test_collect_runs_requires_unique_unchanged_complete_verification(
    tmp_path,
    research_config,
    action: str,
    error: str,
) -> None:
    _canonical, verification = _promoted_winoground_pair(tmp_path, research_config)
    if action == "missing":
        shutil.rmtree(verification)
    elif action == "duplicate":
        shutil.copytree(verification, tmp_path / "duplicate-verification")
    else:
        path = verification / "run.json"
        run = json.loads(path.read_text(encoding="utf-8"))
        run["status"] = "reportable"
        run["review"] = {
            "reviewed_by": "manual",
            "reviewed_at": "2026-07-16T00:00:00+00:00",
            "decision": "reportable",
            "notes": None,
        }
        atomic_write_json(path, run)

    with pytest.raises(ValueError, match=error):
        collect_runs(tmp_path)


@pytest.mark.parametrize(
    ("target", "relative", "error"),
    [
        ("canonical", "evaluation.json", "canonical artifact digest mismatch"),
        ("canonical", "metrics.json", "canonical artifact digest mismatch"),
        ("canonical", "predictions.jsonl", "canonical artifact digest mismatch"),
        ("canonical", "config.resolved.yaml", "canonical artifact digest mismatch"),
        ("verification", "evaluation.json", "verification artifact digest mismatch"),
        ("verification", "predictions.jsonl", "verification artifact digest mismatch"),
        ("verification", "config.resolved.yaml", "verification artifact digest mismatch"),
    ],
)
def test_collect_runs_rejects_tampered_promoted_artifacts(
    tmp_path,
    research_config,
    target: str,
    relative: str,
    error: str,
) -> None:
    canonical, verification = _promoted_winoground_pair(tmp_path, research_config)
    path = (canonical if target == "canonical" else verification) / relative
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(ValueError, match=error):
        collect_runs(tmp_path)


@pytest.mark.parametrize(
    "unsafe_path",
    ["../../comparison.json", "/tmp/review.csv", r"C:\review.csv"],
)
def test_collect_runs_rejects_unsafe_evidence_paths(
    tmp_path, research_config, unsafe_path: str
) -> None:
    canonical, _verification = _promoted_winoground_pair(tmp_path, research_config)
    path = canonical / "run.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    run["review"]["comparison_file"] = unsafe_path
    atomic_write_json(path, run)

    with pytest.raises(ValueError, match="unsafe comparison evidence path"):
        collect_runs(tmp_path)


def _promoted_winoground_pair(tmp_path, research_config):
    canonical = _complete_winoground_run(tmp_path, research_config)
    verification = _complete_winoground_verification_run(tmp_path, research_config)
    promote_run(
        canonical,
        verification_run=verification,
        prediction_review=_write_synthetic_prediction_review(tmp_path),
        reviewed_by="Research Reviewer",
    )
    return canonical, verification


def _update_review_digest(run_dir, stem: str, path) -> None:
    from recoalign.data.manifest import sha256_file

    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["review"][f"{stem}_sha256"] = sha256_file(path)
    atomic_write_json(run_path, run)


def _synchronize_evidence_hash(run_dir, stem: str, path) -> None:
    from recoalign.data.manifest import sha256_file

    evidence_path = run_dir / "review" / "promotion_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence[f"{stem}_sha256"] = sha256_file(path)
    atomic_write_json(evidence_path, evidence)
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["review"][f"{stem}_sha256"] = sha256_file(path)
    run["review"]["promotion_evidence_sha256"] = sha256_file(evidence_path)
    atomic_write_json(run_path, run)
