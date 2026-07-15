import json

import pytest

from recoalign.experiments.records import create_run, finalize_run, promote_run
from recoalign.reproducibility import atomic_write_json
from recoalign.schema_validation import validate_payload


def _set_clean_git(run_dir) -> None:
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
    validate_payload("environment", environment)
    atomic_write_json(environment_path, environment)

    run_path = run_dir / "run.json"
    record = json.loads(run_path.read_text(encoding="utf-8"))
    record["git_commit"] = "a" * 40
    validate_payload("run", record)
    atomic_write_json(run_path, record)


def test_run_lifecycle_snapshots_manifests(tmp_path, research_config) -> None:
    run_dir = create_run(research_config, output_root=tmp_path, run_id="fixed-run")
    assert (run_dir / "config.resolved.yaml").is_file()
    assert (run_dir / "environment.json").is_file()
    assert (run_dir / "manifests" / "dataset.yaml").is_file()
    assert (run_dir / "manifests" / "checkpoint.yaml").is_file()

    record = finalize_run(run_dir, {"R@1": 50, "mean_recall": 75.25}, status="complete")
    assert record["status"] == "complete"
    assert record["dataset_verification"]["status"] == "passed"


def test_finalize_cannot_mark_reportable(tmp_path, research_config) -> None:
    run_dir = create_run(research_config, output_root=tmp_path, run_id="fixed-run")
    with pytest.raises(ValueError, match="promote-run"):
        finalize_run(run_dir, {"R@1": 50}, status="reportable")


def test_promote_run_requires_clean_reviewed_complete_run(tmp_path, research_config) -> None:
    run_dir = create_run(research_config, output_root=tmp_path, run_id="fixed-run")
    finalize_run(run_dir, {"R@1": 50}, status="complete")
    _set_clean_git(run_dir)

    record = promote_run(run_dir, reviewed_by="Research Reviewer", notes="checked")
    assert record["status"] == "reportable"
    assert record["review"]["reviewed_by"] == "Research Reviewer"


def test_promote_rejects_dirty_git(tmp_path, research_config) -> None:
    run_dir = create_run(research_config, output_root=tmp_path, run_id="fixed-run")
    finalize_run(run_dir, {"R@1": 50}, status="complete")
    _set_clean_git(run_dir)

    environment_path = run_dir / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["git"]["dirty"] = True
    environment["git"]["diff_sha256"] = "b" * 64
    atomic_write_json(environment_path, environment)

    with pytest.raises(ValueError, match="dirty"):
        promote_run(run_dir, reviewed_by="Research Reviewer")
