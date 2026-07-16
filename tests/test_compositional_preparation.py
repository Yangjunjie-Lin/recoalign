from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from recoalign.data.compositional_preparation import (
    prepare_aro,
    prepare_bivlc,
    prepare_winoground,
)
from recoalign.data.manifest import load_dataset_manifest, verify_manifest_files


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
