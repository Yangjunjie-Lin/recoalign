import json
from pathlib import Path

import pytest
import yaml

from recoalign.experiments.records import create_run, fail_run, finalize_run, promote_run
from recoalign.reproducibility import atomic_write_json
from recoalign.schema_validation import validate_payload

REVISION = "a" * 40
CONTENT_CHECK_METHOD = "casefolded_alphanumeric_character_multiset_v1"


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


@pytest.mark.parametrize(
    "provenance_status",
    [
        "requires_regeneration_from_pinned_revision",
        "synthetic_or_unverified",
        "template_not_generated",
    ],
)
def test_promote_rejects_non_reportable_winoground_provenance_states(
    tmp_path,
    research_config,
    provenance_status: str,
) -> None:
    run_dir = _complete_winoground_run(
        tmp_path,
        research_config,
        processing_updates={
            "provenance_status": provenance_status,
            "source_revision": None,
        },
    )

    with pytest.raises(ValueError, match="provenance is not pinned and verified"):
        promote_run(run_dir, reviewed_by="Research Reviewer")


def test_promote_rejects_missing_winoground_provenance_status(
    tmp_path, research_config
) -> None:
    run_dir = _complete_winoground_run(
        tmp_path,
        research_config,
        remove_processing=("provenance_status",),
    )

    with pytest.raises(ValueError, match="provenance is not pinned and verified"):
        promote_run(run_dir, reviewed_by="Research Reviewer")


@pytest.mark.parametrize("revision", ["main", "abc123", "a" * 39])
def test_promote_rejects_invalid_winoground_revision(
    tmp_path,
    research_config,
    revision: str,
) -> None:
    run_dir = _complete_winoground_run(
        tmp_path,
        research_config,
        processing_updates={"source_revision": revision},
    )

    with pytest.raises(ValueError, match="40-character commit SHA"):
        promote_run(run_dir, reviewed_by="Research Reviewer")


def test_promote_rejects_missing_winoground_exporter_version(
    tmp_path, research_config
) -> None:
    run_dir = _complete_winoground_run(
        tmp_path,
        research_config,
        remove_processing=("exporter_version",),
    )

    with pytest.raises(ValueError, match="exporter_version is missing"):
        promote_run(run_dir, reviewed_by="Research Reviewer")


@pytest.mark.parametrize("downloaded_at", [None, "not-a-date", "2026-07-16T00:00:00"])
def test_promote_rejects_missing_or_invalid_winoground_downloaded_at(
    tmp_path,
    research_config,
    downloaded_at: str | None,
) -> None:
    run_dir = _complete_winoground_run(
        tmp_path,
        research_config,
        manifest_updates={"downloaded_at": downloaded_at},
    )

    with pytest.raises(ValueError, match="downloaded_at is missing or invalid"):
        promote_run(run_dir, reviewed_by="Research Reviewer")


def test_promote_rejects_noncanonical_winoground_sample_count(
    tmp_path, research_config
) -> None:
    run_dir = _complete_winoground_run(
        tmp_path,
        research_config,
        manifest_updates={"splits": {"test": 1}},
    )

    with pytest.raises(ValueError, match="exactly 400 test samples"):
        promote_run(run_dir, reviewed_by="Research Reviewer")


@pytest.mark.parametrize(
    ("processing_updates", "remove_processing"),
    [
        ({}, ("caption_alphanumeric_character_multiset_match_rate",)),
        ({"caption_alphanumeric_character_multiset_match_rate": 99.75}, ()),
        ({"caption_alphanumeric_character_multiset_method": "wrong-method"}, ()),
    ],
)
def test_promote_rejects_invalid_canonical_winoground_content_check(
    tmp_path,
    research_config,
    processing_updates: dict[str, object],
    remove_processing: tuple[str, ...],
) -> None:
    run_dir = _complete_winoground_run(
        tmp_path,
        research_config,
        processing_updates=processing_updates,
        remove_processing=remove_processing,
    )

    with pytest.raises(ValueError, match="caption content check is missing or invalid"):
        promote_run(run_dir, reviewed_by="Research Reviewer")


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("source_dataset", "local/winoground", "source_dataset must be facebook/winoground"),
        ("source_split", "validation", "source_split must be test"),
        ("image_hashes", False, "require hashed images"),
    ],
)
def test_promote_rejects_invalid_winoground_source_contract(
    tmp_path,
    research_config,
    field: str,
    value: object,
    error: str,
) -> None:
    run_dir = _complete_winoground_run(
        tmp_path,
        research_config,
        processing_updates={field: value},
    )

    with pytest.raises(ValueError, match=error):
        promote_run(run_dir, reviewed_by="Research Reviewer")


def test_complete_pinned_winoground_manifest_passes_promotion_gate_fixture(
    tmp_path, research_config
) -> None:
    run_dir = _complete_winoground_run(tmp_path, research_config)

    record = promote_run(run_dir, reviewed_by="Research Reviewer")

    assert record["status"] == "reportable"


def _complete_winoground_run(
    tmp_path: Path,
    research_config,
    *,
    processing_updates: dict[str, object] | None = None,
    remove_processing: tuple[str, ...] = (),
    manifest_updates: dict[str, object] | None = None,
) -> Path:
    manifest_path = Path(research_config["data"]["manifest"])
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "name": "winoground",
            "downloaded_at": "2026-07-16T00:00:00Z",
            "splits": {"test": 400},
            "processing": {
                "provenance_status": "pinned_revision_verified",
                "source_dataset": "facebook/winoground",
                "source_split": "test",
                "source_revision": REVISION,
                "exporter_version": "winoground-hf-export-v2",
                "image_hashes": True,
                "caption_alphanumeric_character_multiset_match_rate": 100.0,
                "caption_alphanumeric_character_multiset_method": CONTENT_CHECK_METHOD,
            },
        }
    )
    manifest["processing"].update(processing_updates or {})
    for field in remove_processing:
        manifest["processing"].pop(field, None)
    manifest.update(manifest_updates or {})
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    research_config["data"]["dataset"] = "winoground"

    run_dir = create_run(research_config, output_root=tmp_path, run_id="winoground-gate")
    finalize_run(run_dir, {"group_accuracy": 50.0}, status="complete")
    _set_clean_git(run_dir)
    return run_dir
