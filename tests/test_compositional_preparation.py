from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from recoalign.data.compositional_preparation import (
    prepare_aro,
    prepare_bivlc,
    prepare_winoground,
)
from recoalign.data.manifest import load_dataset_manifest, verify_manifest_files

REVISION_A = "a" * 40
REVISION_B = "b" * 40
EXPORTER_VERSION = "winoground-hf-export-v2"
DOWNLOADED_AT = "2026-07-16T00:00:00Z"


def test_prepare_aro_writes_normalized_snapshot_and_hashes(tmp_path: Path) -> None:
    root = tmp_path / "aro"
    _images(root)
    source = tmp_path / "aro_source.jsonl"
    _write_jsonl(
        source,
        [
            {
                "sample_id": "relation:0",
                "image": "a.jpg",
                "captions": ["a correct", "b wrong"],
                "correct_index": 0,
                "subset": "vg_relation",
                "tags": ["spatial"],
            }
        ],
    )
    manifest_path = tmp_path / "aro.yaml"

    manifest = prepare_aro(
        source,
        root,
        manifest_output=manifest_path,
        source="authorized local export",
        license_name="upstream terms",
        hash_images=True,
    )

    assert manifest["splits"] == {"test": 1}
    assert manifest["processing"]["subsets"] == {"vg_relation": 1}
    inventory = (root / "inventories" / "images.jsonl").read_text(encoding="utf-8")
    assert "sha256" in inventory
    loaded = load_dataset_manifest(manifest_path)
    assert verify_manifest_files(root, loaded) == []


def test_prepare_winoground_records_alphanumeric_character_multiset_rate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "winoground"
    _images(root)
    source = tmp_path / "winoground_source.jsonl"
    _write_jsonl(
        source,
        [
            {
                "sample_id": "0",
                "image_0": "a.jpg",
                "image_1": "b.jpg",
                "caption_0": "a red cup",
                "caption_1": "cup red a",
                "tags": ["spatial"],
            }
        ],
    )
    manifest_path = tmp_path / "winoground.yaml"

    manifest = prepare_winoground(
        source,
        root,
        manifest_output=manifest_path,
        source="official dataset export",
        license_name="upstream terms",
    )

    assert (
        manifest["processing"]["caption_alphanumeric_character_multiset_match_rate"]
        == 100.0
    )
    assert (
        manifest["processing"]["caption_alphanumeric_character_multiset_method"]
        == "casefolded_alphanumeric_character_multiset_v1"
    )
    assert manifest["processing"]["caption_token_multiset_match_rate"] == 100.0
    assert manifest["processing"]["provenance_status"] == "synthetic_or_unverified"
    row = json.loads((root / "annotations" / "test.jsonl").read_text(encoding="utf-8"))
    assert row["category"] == "winoground"


def test_prepare_winoground_accepts_character_conserved_official_pair(tmp_path: Path) -> None:
    root = tmp_path / "winoground"
    _images(root)
    source = tmp_path / "winoground_source.jsonl"
    _write_jsonl(
        source,
        [
            {
                "sample_id": "13",
                "image_0": "a.jpg",
                "image_1": "b.jpg",
                "caption_0": "a caterpillar with some plants",
                "caption_1": "a plant with some caterpillars",
                "tags": ["Noun", "Morpheme-Level"],
            }
        ],
    )

    manifest = prepare_winoground(
        source,
        root,
        manifest_output=tmp_path / "winoground.yaml",
        source="official dataset export",
        license_name="upstream terms",
    )

    assert (
        manifest["processing"]["caption_alphanumeric_character_multiset_match_rate"]
        == 100.0
    )
    assert (
        manifest["processing"]["caption_alphanumeric_character_multiset_method"]
        == "casefolded_alphanumeric_character_multiset_v1"
    )
    assert manifest["processing"]["provenance_status"] == "synthetic_or_unverified"


def test_formal_winoground_requires_row_revision(tmp_path: Path) -> None:
    metadata = _formal_metadata()
    metadata.pop("source_revision")

    with pytest.raises(ValueError, match="missing pinned row source_revision"):
        _prepare_winoground_rows(tmp_path, _formal_rows(metadata))


def test_cli_revision_cannot_fill_missing_formal_row_revision(tmp_path: Path) -> None:
    metadata = _formal_metadata()
    metadata.pop("source_revision")

    with pytest.raises(ValueError, match="missing pinned row source_revision"):
        _prepare_winoground_rows(
            tmp_path,
            _formal_rows(metadata),
            source_revision=REVISION_A,
        )


def test_formal_winoground_requires_row_exporter_version(tmp_path: Path) -> None:
    metadata = _formal_metadata()
    metadata.pop("exporter_version")

    with pytest.raises(ValueError, match="missing row exporter_version"):
        _prepare_winoground_rows(tmp_path, _formal_rows(metadata))


def test_formal_winoground_rejects_inconsistent_row_revisions(tmp_path: Path) -> None:
    rows = _formal_rows(_formal_metadata())
    rows[1]["metadata"]["source_revision"] = REVISION_B

    with pytest.raises(ValueError, match=r"disagree on metadata\.source_revision"):
        _prepare_winoground_rows(tmp_path, rows)


def test_formal_winoground_rejects_inconsistent_exporter_versions(tmp_path: Path) -> None:
    rows = _formal_rows(_formal_metadata())
    rows[1]["metadata"]["exporter_version"] = "different-exporter"

    with pytest.raises(ValueError, match=r"disagree on metadata\.exporter_version"):
        _prepare_winoground_rows(tmp_path, rows)


def test_formal_winoground_rejects_cli_revision_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source_revision does not match"):
        _prepare_winoground_rows(
            tmp_path,
            _formal_rows(_formal_metadata()),
            source_revision=REVISION_B,
        )


def test_formal_winoground_rejects_cli_exporter_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exporter_version does not match"):
        _prepare_winoground_rows(
            tmp_path,
            _formal_rows(_formal_metadata()),
            exporter_version="different-exporter",
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("source_dataset", "local/winoground", "source_dataset must be facebook/winoground"),
        ("source_split", "validation", "source_split must be test"),
    ],
)
def test_formal_winoground_requires_official_source_identity(
    tmp_path: Path,
    field: str,
    value: str,
    error: str,
) -> None:
    metadata = _formal_metadata()
    metadata[field] = value

    with pytest.raises(ValueError, match=error):
        _prepare_winoground_rows(tmp_path, _formal_rows(metadata))


def test_formal_winoground_requires_downloaded_at(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires downloaded_at"):
        _prepare_winoground_rows(
            tmp_path,
            _formal_rows(_formal_metadata()),
            downloaded_at=None,
        )


def test_pinned_formal_winoground_records_verified_provenance(tmp_path: Path) -> None:
    manifest = _prepare_winoground_rows(tmp_path, _formal_rows(_formal_metadata()))

    assert manifest["splits"] == {"test": 400}
    assert manifest["downloaded_at"] == DOWNLOADED_AT
    assert manifest["processing"]["provenance_status"] == "pinned_revision_verified"
    assert manifest["processing"]["source_dataset"] == "facebook/winoground"
    assert manifest["processing"]["source_split"] == "test"
    assert manifest["processing"]["source_revision"] == REVISION_A
    assert manifest["processing"]["exporter_version"] == EXPORTER_VERSION


def test_small_winoground_with_explicit_provenance_remains_unverified(
    tmp_path: Path,
) -> None:
    manifest = _prepare_winoground_rows(
        tmp_path,
        _formal_rows({})[:1],
        source_revision=REVISION_A,
        exporter_version=EXPORTER_VERSION,
    )

    assert manifest["processing"]["source_revision"] == REVISION_A
    assert manifest["processing"]["provenance_status"] == "synthetic_or_unverified"


def test_prepare_bivlc_preserves_categories(tmp_path: Path) -> None:
    root = tmp_path / "bivlc"
    _images(root)
    source = tmp_path / "bivlc_source.jsonl"
    _write_jsonl(
        source,
        [
            {
                "sample_id": "0",
                "image_0": "a.jpg",
                "image_1": "b.jpg",
                "caption_0": "a positive",
                "caption_1": "b negative",
                "category": "replace_obj",
            }
        ],
    )
    manifest_path = tmp_path / "bivlc.yaml"

    manifest = prepare_bivlc(
        source,
        root,
        manifest_output=manifest_path,
        source="official dataset export",
        license_name="upstream terms",
    )

    assert manifest["processing"]["categories"] == {"replace_obj": 1}
    assert manifest["splits"] == {"test": 1}


def _images(root: Path) -> None:
    image_root = root / "images"
    image_root.mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(image_root / "a.jpg")
    Image.new("RGB", (2, 2)).save(image_root / "b.jpg")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _formal_metadata() -> dict[str, object]:
    return {
        "source_dataset": "facebook/winoground",
        "source_split": "test",
        "source_revision": REVISION_A,
        "exporter_version": EXPORTER_VERSION,
    }


def _formal_rows(metadata: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "sample_id": str(index),
            "image_0": "a.jpg",
            "image_1": "b.jpg",
            "caption_0": "a red cup",
            "caption_1": "cup red a",
            "tags": ["synthetic-fixture"],
            "metadata": deepcopy(metadata),
        }
        for index in range(400)
    ]


def _prepare_winoground_rows(
    tmp_path: Path,
    rows: list[dict[str, object]],
    *,
    source_revision: str | None = None,
    exporter_version: str | None = None,
    downloaded_at: str | None = DOWNLOADED_AT,
) -> dict[str, object]:
    root = tmp_path / "winoground"
    _images(root)
    source = tmp_path / "winoground_source.jsonl"
    _write_jsonl(source, rows)
    return prepare_winoground(
        source,
        root,
        manifest_output=tmp_path / "winoground.yaml",
        source="test fixture",
        license_name="test only",
        source_revision=source_revision,
        exporter_version=exporter_version,
        downloaded_at=downloaded_at,
    )
