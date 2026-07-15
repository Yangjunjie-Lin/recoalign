import json
from pathlib import Path

import pytest
import yaml

from recoalign.experiments.records import create_run, fail_run, finalize_run, promote_run
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


def test_failed_run_is_retained_without_fake_metrics(tmp_path, research_config) -> None:
    run_dir = create_run(research_config, output_root=tmp_path, run_id="fixed-run")
    record = fail_run(run_dir, RuntimeError("model download failed"))
    assert record["status"] == "failed"
    assert record["metrics_file"] is None
    assert "model download failed" in record["notes"]
    assert not (run_dir / "metrics.json").exists()


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


def test_promote_requires_hashed_image_inventory(tmp_path, research_config) -> None:
    dataset_root = Path(research_config["data"]["root"])
    inventory = dataset_root / "images.jsonl"
    sample = dataset_root / "sample.txt"
    inventory.write_text(
        json.dumps({"path": "sample.txt", "bytes": sample.stat().st_size}) + "\n",
        encoding="utf-8",
    )
    manifest_path = Path(research_config["data"]["manifest"])
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["processing"] = {
        "image_inventory": "images.jsonl",
        "image_hashes": False,
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    run_dir = create_run(research_config, output_root=tmp_path, run_id="fixed-run")
    finalize_run(run_dir, {"R@1": 50}, status="complete")
    _set_clean_git(run_dir)
    with pytest.raises(ValueError, match="SHA-256 image inventory"):
        promote_run(run_dir, reviewed_by="Research Reviewer")
