import json
from pathlib import Path

import pytest
import yaml

from recoalign.data.manifest import sha256_file
from recoalign.experiments.records import (
    _validate_prediction_review,
    create_run,
    fail_run,
    finalize_run,
    promote_run,
    validate_completed_winoground_run_integrity,
)
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
    with pytest.raises(ValueError, match="require image_hashes=true"):
        promote_run(run_dir, reviewed_by="Research Reviewer")


def test_promote_rejects_winoground_image_hash_flag_without_inventory(
    tmp_path, research_config
) -> None:
    run_dir = _complete_winoground_run(
        tmp_path,
        research_config,
        remove_processing=("image_inventory",),
    )

    with pytest.raises(ValueError, match="require an image inventory"):
        promote_run(run_dir, reviewed_by="Research Reviewer")


def test_promote_rejects_undeclared_winoground_image_inventory(tmp_path, research_config) -> None:
    run_dir = _complete_winoground_run(
        tmp_path,
        research_config,
        declare_inventory=False,
    )

    with pytest.raises(ValueError, match="not declared in manifest files"):
        promote_run(run_dir, reviewed_by="Research Reviewer")


def test_promote_rejects_winoground_inventory_row_without_sha256(tmp_path, research_config) -> None:
    run_dir = _complete_winoground_run(
        tmp_path,
        research_config,
        remove_inventory_row=("sha256",),
    )

    with pytest.raises(ValueError, match="row missing sha256"):
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


def test_promote_rejects_missing_winoground_provenance_status(tmp_path, research_config) -> None:
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


def test_promote_rejects_missing_winoground_exporter_version(tmp_path, research_config) -> None:
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


def test_promote_rejects_noncanonical_winoground_sample_count(tmp_path, research_config) -> None:
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
        ("image_hashes", False, "require image_hashes=true"),
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
    verification_run = _complete_winoground_verification_run(tmp_path, research_config)
    prediction_review = _write_synthetic_prediction_review(tmp_path)

    record = promote_run(
        run_dir,
        verification_run=verification_run,
        prediction_review=prediction_review,
        reviewed_by="Research Reviewer",
    )

    assert record["status"] == "reportable"
    assert record["review"]["verification_run_id"] == "winoground-verification"
    assert (run_dir / "review" / "promotion_evidence.json").is_file()
    evidence = json.loads((run_dir / "review" / "promotion_evidence.json").read_text())
    assert evidence["comparison_passed"] is True
    assert evidence["prediction_review_count"] == 400
    assert evidence["mapping_checked_count"] == 400
    assert record["review"]["comparison_sha256"] == sha256_file(
        run_dir / "review" / "run_comparison.json"
    )
    assert record["review"]["prediction_review_sha256"] == sha256_file(
        run_dir / "review" / "prediction_review.csv"
    )
    assert record["review"]["promotion_evidence_sha256"] == sha256_file(
        run_dir / "review" / "promotion_evidence.json"
    )
    assert json.loads((verification_run / "run.json").read_text())["status"] == "complete"


def test_winoground_promotion_requires_verification_run(tmp_path, research_config) -> None:
    run_dir = _complete_winoground_run(tmp_path, research_config)

    with pytest.raises(ValueError, match="requires a cache-disabled verification run"):
        promote_run(
            run_dir,
            prediction_review=_write_synthetic_prediction_review(tmp_path),
            reviewed_by="Research Reviewer",
        )
    assert json.loads((run_dir / "run.json").read_text())["status"] == "complete"


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("dataset", "other", "evaluation dataset does not match run"),
        ("split", "validation", "evaluation split does not match run"),
        (
            "dataset_manifest_sha256",
            "b" * 64,
            "evaluation dataset manifest digest does not match run",
        ),
        ("encoder.model", "other", "evaluation encoder model does not match run"),
        (
            "encoder.pretrained",
            "other",
            "evaluation encoder pretrained weights do not match run",
        ),
    ],
)
def test_winoground_promotion_cross_checks_evaluation_identity(
    tmp_path,
    research_config,
    field: str,
    value: str,
    error: str,
) -> None:
    run_dir = _complete_winoground_run(tmp_path, research_config)
    path = run_dir / "evaluation.json"
    evaluation = json.loads(path.read_text())
    if field.startswith("encoder."):
        evaluation["metadata"]["encoder"][field.split(".", maxsplit=1)[1]] = value
    else:
        evaluation["metadata"][field] = value
    atomic_write_json(path, evaluation)

    with pytest.raises(ValueError, match=error):
        promote_run(run_dir, reviewed_by="Research Reviewer")


def test_winoground_promotion_rejects_same_verification_directory(
    tmp_path, research_config
) -> None:
    run_dir = _complete_winoground_run(tmp_path, research_config)

    with pytest.raises(ValueError, match="different from canonical"):
        promote_run(
            run_dir,
            verification_run=run_dir,
            prediction_review=_write_synthetic_prediction_review(tmp_path),
            reviewed_by="Research Reviewer",
        )


def test_winoground_promotion_rejects_existing_evidence(tmp_path, research_config) -> None:
    run_dir = _complete_winoground_run(tmp_path, research_config)
    verification = _complete_winoground_verification_run(tmp_path, research_config)
    (run_dir / "review").mkdir()

    with pytest.raises(ValueError, match="promotion evidence already exists"):
        promote_run(
            run_dir,
            verification_run=verification,
            prediction_review=_write_synthetic_prediction_review(tmp_path),
            reviewed_by="Research Reviewer",
        )
    assert json.loads((run_dir / "run.json").read_text())["status"] == "complete"


def test_failed_promotion_leaves_no_staged_evidence(tmp_path, research_config) -> None:
    run_dir = _complete_winoground_run(tmp_path, research_config)
    verification = _complete_winoground_verification_run(tmp_path, research_config)
    review = _write_synthetic_prediction_review(tmp_path)
    review.write_text(review.read_text().replace(",true,", ",false,", 1), encoding="utf-8")

    with pytest.raises(ValueError, match="mapping_checked=true"):
        promote_run(
            run_dir,
            verification_run=verification,
            prediction_review=review,
            reviewed_by="Research Reviewer",
        )
    assert not (run_dir / "review").exists()
    assert not list(run_dir.glob(".review-staging-*"))
    assert json.loads((run_dir / "run.json").read_text())["status"] == "complete"


@pytest.mark.parametrize("status", ["failed", "reportable"])
def test_winoground_promotion_requires_complete_verification_status(
    tmp_path, research_config, status: str
) -> None:
    run_dir = _complete_winoground_run(tmp_path, research_config)
    verification = _complete_winoground_verification_run(tmp_path, research_config)
    updates: dict[str, object] = {"status": status}
    if status == "reportable":
        updates["review"] = {
            "reviewed_by": "synthetic reviewer",
                "reviewed_at": "2026-07-16T00:00:00+00:00",
                "decision": "reportable",
                "notes": "synthetic fixture",
                "verification_run_id": "other",
                "promotion_evidence_file": "review/promotion_evidence.json",
                "promotion_evidence_sha256": "a" * 64,
                "comparison_file": "review/run_comparison.json",
                "comparison_sha256": "a" * 64,
                "prediction_review_file": "review/prediction_review.csv",
                "prediction_review_sha256": "a" * 64,
                "prediction_review_count": 400,
            }
    _update_run_record(verification, **updates)

    with pytest.raises(ValueError, match="requires status complete"):
        promote_run(
            run_dir,
            verification_run=verification,
            prediction_review=_write_synthetic_prediction_review(tmp_path),
            reviewed_by="Research Reviewer",
        )


def test_winoground_promotion_requires_null_verification_review(tmp_path, research_config) -> None:
    run_dir = _complete_winoground_run(tmp_path, research_config)
    verification = _complete_winoground_verification_run(tmp_path, research_config)
    _update_run_record(
        verification,
        review={
            "reviewed_by": "synthetic reviewer",
            "reviewed_at": "2026-07-16T00:00:00+00:00",
            "decision": "reportable",
            "notes": "synthetic fixture",
        },
    )

    with pytest.raises(ValueError, match="requires review null"):
        promote_run(
            run_dir,
            verification_run=verification,
            prediction_review=_write_synthetic_prediction_review(tmp_path),
            reviewed_by="Research Reviewer",
        )


@pytest.mark.parametrize("target", ["canonical", "verification"])
def test_winoground_promotion_rejects_invalid_cache_pair(
    tmp_path, research_config, target: str
) -> None:
    run_dir = _complete_winoground_run(tmp_path, research_config)
    verification = _complete_winoground_verification_run(tmp_path, research_config)
    changed = run_dir if target == "canonical" else verification
    evaluation_path = changed / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text())
    evaluation["metadata"]["cache"]["enabled"] = target == "verification"
    atomic_write_json(evaluation_path, evaluation)

    with pytest.raises(ValueError, match="run cache must be"):
        promote_run(
            run_dir,
            verification_run=verification,
            prediction_review=_write_synthetic_prediction_review(tmp_path),
            reviewed_by="Research Reviewer",
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("config_sha256", "b" * 64, "config digest"),
        ("git_commit", "b" * 40, "environment.json"),
        ("seed", 999, "seed does not match"),
        ("dataset_manifest_sha256", "b" * 64, "dataset manifest digest"),
    ],
)
def test_winoground_promotion_recomputes_identity_comparison(
    tmp_path, research_config, field: str, value: object, error: str
) -> None:
    run_dir = _complete_winoground_run(tmp_path, research_config)
    verification = _complete_winoground_verification_run(tmp_path, research_config)
    _update_run_record(verification, **{field: value})

    with pytest.raises(ValueError, match=error):
        promote_run(
            run_dir,
            verification_run=verification,
            prediction_review=_write_synthetic_prediction_review(tmp_path),
            reviewed_by="Research Reviewer",
        )


@pytest.mark.parametrize("mutation", ["score", "decision"])
def test_winoground_promotion_recomputes_prediction_comparison(
    tmp_path, research_config, mutation: str
) -> None:
    run_dir = _complete_winoground_run(tmp_path, research_config)
    verification = _complete_winoground_verification_run(tmp_path, research_config)
    path = verification / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "score":
        rows[0]["scores"][0] += 0.1
    else:
        rows[0]["tie"] = True
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    error = "comparison did not pass" if mutation == "score" else "does not match recomputed scores"
    with pytest.raises(ValueError, match=error):
        promote_run(
            run_dir,
            verification_run=verification,
            prediction_review=_write_synthetic_prediction_review(tmp_path),
            reviewed_by="Research Reviewer",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model.name", "other"),
        ("model.pretrained", "other"),
        ("experiment.seed", 999),
        ("model.precision", "fp16"),
        ("data.root", "other-root"),
    ],
)
def test_verification_integrity_rejects_tampered_resolved_config(
    tmp_path: Path,
    research_config,
    field: str,
    value: object,
) -> None:
    _complete_winoground_run(tmp_path, research_config)
    verification = _complete_winoground_verification_run(tmp_path, research_config)
    config_path = verification / "config.resolved.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    section, name = field.split(".")
    config[section][name] = value
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    run_field = {
        "model.name": "model",
        "model.pretrained": "pretrained",
        "experiment.seed": "seed",
        "model.precision": "precision",
    }.get(field)
    if run_field is not None:
        _update_run_record(verification, **{run_field: value})

    with pytest.raises(ValueError, match="config digest"):
        validate_completed_winoground_run_integrity(
            verification,
            expected_cache_enabled=False,
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("commit", "b" * 40, "inconsistent with environment"),
        ("dirty", True, "dirty or unknown"),
        ("untracked_count", 1, "untracked files"),
    ],
)
def test_verification_integrity_rejects_tampered_environment(
    tmp_path: Path,
    research_config,
    field: str,
    value: object,
    error: str,
) -> None:
    _complete_winoground_run(tmp_path, research_config)
    verification = _complete_winoground_verification_run(tmp_path, research_config)
    path = verification / "environment.json"
    environment = json.loads(path.read_text(encoding="utf-8"))
    environment["git"][field] = value
    atomic_write_json(path, environment)

    with pytest.raises(ValueError, match=error):
        validate_completed_winoground_run_integrity(
            verification,
            expected_cache_enabled=False,
        )


@pytest.mark.parametrize("manifest", ["dataset", "checkpoint"])
def test_verification_integrity_rejects_tampered_manifest_snapshot(
    tmp_path: Path,
    research_config,
    manifest: str,
) -> None:
    _complete_winoground_run(tmp_path, research_config)
    verification = _complete_winoground_verification_run(tmp_path, research_config)
    path = verification / "manifests" / f"{manifest}.yaml"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=f"{manifest} manifest digest"):
        validate_completed_winoground_run_integrity(
            verification,
            expected_cache_enabled=False,
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("dataset", "other", "dataset does not match"),
        ("split", "validation", "split does not match"),
        ("dataset_manifest_sha256", "b" * 64, "manifest digest"),
        ("encoder.model", "other", "encoder model"),
        ("encoder.pretrained", "other", "pretrained weights"),
        ("encoder.precision", "fp16", "encoder precision"),
        ("annotation_sha256", "b" * 64, "annotation sha256"),
    ],
)
def test_verification_integrity_rejects_tampered_evaluation_identity(
    tmp_path: Path,
    research_config,
    field: str,
    value: object,
    error: str,
) -> None:
    _complete_winoground_run(tmp_path, research_config)
    verification = _complete_winoground_verification_run(tmp_path, research_config)
    path = verification / "evaluation.json"
    evaluation = json.loads(path.read_text(encoding="utf-8"))
    if field.startswith("encoder."):
        evaluation["metadata"]["encoder"][field.split(".", maxsplit=1)[1]] = value
    else:
        evaluation["metadata"][field] = value
    atomic_write_json(path, evaluation)

    with pytest.raises(ValueError, match=error):
        validate_completed_winoground_run_integrity(
            verification,
            expected_cache_enabled=False,
        )


def test_verification_integrity_audits_decisions_independently(
    tmp_path: Path, research_config
) -> None:
    canonical = _complete_winoground_run(tmp_path, research_config)
    verification = _complete_winoground_verification_run(tmp_path, research_config)
    for run_dir in (canonical, verification):
        path = run_dir / "predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["group_correct"] = False
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match recomputed scores"):
        validate_completed_winoground_run_integrity(
            verification,
            expected_cache_enabled=False,
        )


def test_verification_integrity_recomputes_metrics_from_scores(
    tmp_path: Path, research_config
) -> None:
    _complete_winoground_run(tmp_path, research_config)
    verification = _complete_winoground_verification_run(tmp_path, research_config)
    metrics_path = verification / "metrics.json"
    evaluation_path = verification / "evaluation.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["group_accuracy/winoground"] = 0.0
    atomic_write_json(metrics_path, metrics)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["metrics"] = metrics
    atomic_write_json(evaluation_path, evaluation)

    with pytest.raises(ValueError, match="does not match recomputed predictions"):
        validate_completed_winoground_run_integrity(
            verification,
            expected_cache_enabled=False,
        )


def test_synthetic_promotion_evidence_fixture_review_validation(tmp_path: Path) -> None:
    prediction_ids = [f"winoground-{index:06d}" for index in range(400)]
    summary = _validate_prediction_review(
        _write_synthetic_prediction_review(tmp_path), prediction_ids
    )

    assert summary["prediction_review_count"] == 400
    assert summary["mapping_checked_count"] == 400
    assert summary["visual_review_status_counts"]["pass"] == 400


@pytest.mark.parametrize(
    ("column", "value", "error"),
    [
        ("mapping_checked", "false", "mapping_checked=true"),
        ("mapping_checked", "yes", "must be true or false"),
        ("visual_review_status", "done", "visual_review_status is invalid"),
        ("annotation_issue", "maybe", "annotation_issue is invalid"),
        ("visual_review_status", "issue", "issue rows require notes"),
        ("visual_review_status", "uncertain", "uncertain rows require notes"),
    ],
)
def test_prediction_review_rejects_invalid_rows(
    tmp_path: Path,
    column: str,
    value: str,
    error: str,
) -> None:
    path = tmp_path / "review.csv"
    row = {
        "sample_id": "sample-0",
        "review_group": "correct",
        "mapping_checked": "true",
        "visual_review_status": "pass",
        "annotation_issue": "none",
        "notes": "",
    }
    row[column] = value
    review_header = list(row)
    path.write_text(
        ",".join(review_header) + "\n" + ",".join(row[field] for field in review_header) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=error):
        _validate_prediction_review(path, ["sample-0"])


def test_prediction_review_rejects_missing_duplicate_and_unknown_ids(tmp_path: Path) -> None:
    header = "sample_id,review_group,mapping_checked,visual_review_status,annotation_issue,notes\n"
    valid = "sample-0,correct,true,pass,none,synthetic\n"
    unknown = "sample-x,other,true,pass,none,synthetic\n"
    cases = {
        "missing": (header + valid, ["sample-0", "sample-1"], "missing 1"),
        "duplicate": (header + valid + valid, ["sample-0"], "duplicate sample_id"),
        "unknown": (header + valid + unknown, ["sample-0"], "unknown sample IDs"),
    }
    for name, (content, expected, error) in cases.items():
        path = tmp_path / f"{name}.csv"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match=error):
            _validate_prediction_review(path, expected)


def test_prediction_review_requires_file_and_complete_header(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="CSV is missing"):
        _validate_prediction_review(tmp_path / "missing.csv", ["sample-0"])

    path = tmp_path / "incomplete.csv"
    path.write_text("sample_id,mapping_checked\nsample-0,true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required headers"):
        _validate_prediction_review(path, ["sample-0"])


def test_create_run_rejects_config_dataset_manifest_mismatch(tmp_path, research_config) -> None:
    research_config["data"]["dataset"] = "winoground"

    with pytest.raises(
        ValueError,
        match="dataset identity mismatch: config declares winoground but manifest declares toy",
    ):
        create_run(research_config, output_root=tmp_path, run_id="identity-mismatch")


def test_create_run_rejects_split_missing_from_manifest(tmp_path, research_config) -> None:
    research_config["data"]["split"] = "validation"

    with pytest.raises(
        ValueError,
        match="split validation is not declared by manifest toy",
    ):
        create_run(research_config, output_root=tmp_path, run_id="split-mismatch")


def test_promote_rejects_tampered_run_dataset(tmp_path, research_config) -> None:
    run_dir = _complete_toy_run(tmp_path, research_config, "tampered-run-dataset")
    _update_run_record(run_dir, dataset="synthetic")

    with pytest.raises(ValueError, match="run dataset does not match resolved config"):
        promote_run(run_dir, reviewed_by="Research Reviewer")


def test_promote_rejects_tampered_resolved_config_dataset(tmp_path, research_config) -> None:
    run_dir = _complete_toy_run(tmp_path, research_config, "tampered-config-dataset")
    config_path = run_dir / "config.resolved.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"]["dataset"] = "synthetic"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="run config digest does not match resolved config"):
        promote_run(run_dir, reviewed_by="Research Reviewer")


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("model", "name", "tampered-model"),
        ("model", "pretrained", "tampered-weights"),
        ("experiment", "seed", 999),
        ("model", "precision", "fp16"),
        ("model", "checkpoint", "tampered-checkpoint"),
    ],
)
def test_promote_rejects_tampered_resolved_config_digest_for_non_dataset_fields(
    tmp_path,
    research_config,
    section: str,
    field: str,
    value: object,
) -> None:
    run_dir = _complete_toy_run(tmp_path, research_config, f"tampered-config-{field}")
    config_path = run_dir / "config.resolved.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config[section][field] = value
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="run config digest does not match resolved config"):
        promote_run(run_dir, reviewed_by="Research Reviewer")


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("model", "tampered-model", "run model does not match resolved config"),
        (
            "pretrained",
            "tampered-weights",
            "run pretrained weights do not match resolved config",
        ),
        ("seed", 999, "run seed does not match resolved config"),
        ("precision", "fp16", "run precision does not match resolved config"),
        (
            "checkpoint",
            "tampered-checkpoint",
            "run checkpoint does not match resolved config",
        ),
    ],
)
def test_promote_rejects_tampered_run_config_identity(
    tmp_path,
    research_config,
    field: str,
    value: object,
    error: str,
) -> None:
    run_dir = _complete_toy_run(tmp_path, research_config, f"tampered-run-{field}")
    _update_run_record(run_dir, **{field: value})

    with pytest.raises(ValueError, match=error):
        promote_run(run_dir, reviewed_by="Research Reviewer")


def test_promote_rejects_tampered_snapshot_manifest_name(tmp_path, research_config) -> None:
    run_dir = _complete_toy_run(tmp_path, research_config, "tampered-manifest-name")
    manifest_path = run_dir / "manifests" / "dataset.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["name"] = "synthetic"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest digest does not match snapshotted manifest"):
        promote_run(run_dir, reviewed_by="Research Reviewer")


def test_promote_rejects_tampered_run_split(tmp_path, research_config) -> None:
    run_dir = _complete_toy_run(tmp_path, research_config, "tampered-run-split")
    _update_run_record(run_dir, dataset_split="validation")

    with pytest.raises(ValueError, match="run dataset split does not match resolved config"):
        promote_run(run_dir, reviewed_by="Research Reviewer")


def test_promote_rejects_snapshot_manifest_without_run_split(tmp_path, research_config) -> None:
    run_dir = _complete_toy_run(tmp_path, research_config, "missing-manifest-split")
    manifest_path = run_dir / "manifests" / "dataset.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["splits"] = {"validation": 1}
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest digest does not match snapshotted manifest"):
        promote_run(run_dir, reviewed_by="Research Reviewer")


def test_promote_rejects_tampered_snapshot_manifest_digest(tmp_path, research_config) -> None:
    run_dir = _complete_toy_run(tmp_path, research_config, "tampered-manifest-digest")
    manifest_path = run_dir / "manifests" / "dataset.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["notes"] = "tampered after run creation"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="digest does not match snapshotted manifest"):
        promote_run(run_dir, reviewed_by="Research Reviewer")


def _complete_winoground_run(
    tmp_path: Path,
    research_config,
    *,
    processing_updates: dict[str, object] | None = None,
    remove_processing: tuple[str, ...] = (),
    manifest_updates: dict[str, object] | None = None,
    declare_inventory: bool = True,
    remove_inventory_row: tuple[str, ...] = (),
) -> Path:
    manifest_path = Path(research_config["data"]["manifest"])
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    dataset_root = Path(research_config["data"]["root"])
    image_path = dataset_root / "images" / "synthetic-image.bin"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"synthetic-image-bytes")
    inventory_row = {
        "path": "images/synthetic-image.bin",
        "bytes": image_path.stat().st_size,
        "sha256": sha256_file(image_path),
    }
    for field in remove_inventory_row:
        inventory_row.pop(field, None)
    inventory_path = dataset_root / "inventories" / "images.jsonl"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory_row) + "\n", encoding="utf-8")
    inventory_entry = {
        "path": "inventories/images.jsonl",
        "bytes": inventory_path.stat().st_size,
        "sha256": sha256_file(inventory_path),
    }
    if declare_inventory:
        manifest["files"].append(inventory_entry)
    annotation_path = dataset_root / "annotations" / "test.jsonl"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_rows = [
        {
            "sample_id": f"winoground-{index:06d}",
            "image_0": "synthetic-image.bin",
            "image_1": "synthetic-image.bin",
            "caption_0": "synthetic caption",
            "caption_1": "caption synthetic",
            "category": "winoground",
            "tags": ["synthetic"],
            "metadata": {},
        }
        for index in range(400)
    ]
    annotation_path.write_text(
        "".join(json.dumps(row) + "\n" for row in annotation_rows),
        encoding="utf-8",
    )
    manifest["files"].append(
        {
            "path": "annotations/test.jsonl",
            "bytes": annotation_path.stat().st_size,
            "sha256": sha256_file(annotation_path),
        }
    )
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
                "image_inventory": "inventories/images.jsonl",
                "image_hashes": True,
                "format": "recoalign-paired-matrix-jsonl-v1",
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
    research_config["data"]["annotation_file"] = str(annotation_path)
    research_config["data"]["image_root"] = str(dataset_root / "images")

    run_dir = create_run(research_config, output_root=tmp_path, run_id="winoground-gate")
    _write_synthetic_winoground_outputs(run_dir, cache_enabled=True)
    _set_clean_git(run_dir)
    return run_dir


def _complete_winoground_verification_run(
    tmp_path: Path,
    research_config,
) -> Path:
    run_dir = create_run(
        research_config,
        output_root=tmp_path,
        run_id="winoground-verification",
    )
    _write_synthetic_winoground_outputs(run_dir, cache_enabled=False)
    _set_clean_git(run_dir)
    return run_dir


def _write_synthetic_winoground_outputs(run_dir: Path, *, cache_enabled: bool) -> None:
    metrics = {
        "image_to_text_accuracy": 100.0,
        "text_to_image_accuracy": 100.0,
        "group_accuracy": 100.0,
        "tie_rate": 0.0,
        "mean_image_to_text_margin": 0.6,
        "mean_text_to_image_margin": 0.7,
        "image_to_text_accuracy/winoground": 100.0,
        "text_to_image_accuracy/winoground": 100.0,
        "group_accuracy/winoground": 100.0,
        "macro_category_group_accuracy": 100.0,
        "group_accuracy/tag/synthetic": 100.0,
        "caption_alphanumeric_character_multiset_match_rate": 100.0,
        "caption_token_multiset_match_rate": 100.0,
    }
    finalize_run(run_dir, metrics, status="complete")
    predictions = [
        {
            "sample_id": f"winoground-{index:06d}",
            "category": "winoground",
            "tags": ["synthetic"],
            "scores": [0.9, 0.1, 0.2, 0.8],
            "image_to_text_correct": True,
            "text_to_image_correct": True,
            "group_correct": True,
            "tie": False,
        }
        for index in range(400)
    ]
    (run_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions),
        encoding="utf-8",
    )
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    annotation_path = (
        Path(
            yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))["data"][
                "root"
            ]
        )
        / "annotations"
        / "test.jsonl"
    )
    evaluation = {
        "schema_version": 1,
        "created_at": "2026-07-16T00:00:00+00:00",
        "metrics": metrics,
        "metadata": {
            "protocol_version": 1,
            "protocol": "winoground-official-400-v1",
            "dataset": "winoground",
            "split": "test",
            "annotation_sha256": sha256_file(annotation_path),
            "dataset_manifest_sha256": record["dataset_manifest_sha256"],
            "encoder": {
                "framework": "synthetic",
                "model": record["model"],
                "pretrained": record["pretrained"],
                "precision": record["precision"],
            },
            "benchmark": "winoground_paired_matrix",
            "num_samples": 400,
            "cache": {
                "enabled": cache_enabled,
                "images_hit": False,
                "texts_hit": False,
            },
        },
        "predictions_file": "predictions.jsonl",
    }
    atomic_write_json(run_dir / "evaluation.json", evaluation)


def _write_synthetic_prediction_review(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic-promotion-evidence-fixture.csv"
    rows = ["sample_id,review_group,mapping_checked,visual_review_status,annotation_issue,notes"]
    rows.extend(
        f"winoground-{index:06d},correct,true,pass,none,synthetic fixture" for index in range(400)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _complete_toy_run(tmp_path: Path, research_config, run_id: str) -> Path:
    run_dir = create_run(research_config, output_root=tmp_path, run_id=run_id)
    finalize_run(run_dir, {"R@1": 50.0}, status="complete")
    _set_clean_git(run_dir)
    return run_dir


def _update_run_record(run_dir: Path, **updates: object) -> None:
    run_path = run_dir / "run.json"
    record = json.loads(run_path.read_text(encoding="utf-8"))
    record.update(updates)
    validate_payload("run", record)
    atomic_write_json(run_path, record)
