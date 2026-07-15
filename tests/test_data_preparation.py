from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from recoalign.data.manifest import load_dataset_manifest, verify_dataset
from recoalign.data.preparation import (
    SUGARCREPE_CATEGORIES,
    prepare_coco,
    prepare_flickr30k,
    prepare_sugarcrepe,
)


def test_prepare_flickr30k_and_verify_inventory(tmp_path: Path) -> None:
    root = tmp_path / "flickr30k"
    images = root / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(images / "a.jpg")
    source = tmp_path / "dataset_flickr30k.json"
    source.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "imgid": 1,
                        "filename": "a.jpg",
                        "split": "test",
                        "sentences": [{"raw": "A small test image."}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "flickr.yaml"
    prepare_flickr30k(
        source,
        root,
        manifest_output=manifest_path,
        source="local fixture",
        license_name="test only",
    )
    manifest = load_dataset_manifest(manifest_path)
    assert manifest["splits"] == {"train": 0, "validation": 0, "test": 1}
    assert verify_dataset(root, manifest) == []
    (images / "a.jpg").write_bytes(b"changed")
    failures = verify_dataset(root, manifest)
    assert any("size mismatch: images/a.jpg" in failure for failure in failures)


def test_prepare_all_sugarcrepe_categories(tmp_path: Path) -> None:
    root = tmp_path / "sugarcrepe"
    images = root / "images/val2017"
    images.mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(images / "a.jpg")
    official = tmp_path / "official"
    official.mkdir()
    for category in SUGARCREPE_CATEGORIES:
        (official / f"{category}.json").write_text(
            json.dumps(
                {
                    "0": {
                        "filename": "a.jpg",
                        "caption": "A correct caption.",
                        "negative_caption": "An incorrect caption.",
                    }
                }
            ),
            encoding="utf-8",
        )
    manifest_path = tmp_path / "sugar.yaml"
    manifest = prepare_sugarcrepe(
        official,
        root,
        manifest_output=manifest_path,
        source="official fixture",
        license_name="test only",
    )
    assert manifest["splits"] == {"test": len(SUGARCREPE_CATEGORIES)}
    assert verify_dataset(root, load_dataset_manifest(manifest_path)) == []
    rows = (root / "annotations/test.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == len(SUGARCREPE_CATEGORIES)


def test_preparation_rejects_missing_provenance(tmp_path: Path) -> None:
    root = tmp_path / "flickr30k"
    images = root / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(images / "a.jpg")
    source = tmp_path / "dataset_flickr30k.json"
    source.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "imgid": 1,
                        "filename": "a.jpg",
                        "split": "test",
                        "sentences": [{"raw": "A test image."}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source"):
        prepare_flickr30k(
            source,
            root,
            manifest_output=tmp_path / "manifest.yaml",
            source=" ",
            license_name="test only",
        )


def test_prepare_coco_karpathy_split(tmp_path: Path) -> None:
    root = tmp_path / "coco"
    images = root / "images/val2014"
    images.mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(images / "COCO_val2014_000000000001.jpg")
    source = tmp_path / "dataset_coco.json"
    source.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "cocoid": 1,
                        "filepath": "val2014",
                        "filename": "COCO_val2014_000000000001.jpg",
                        "split": "test",
                        "sentences": [{"raw": "A small COCO test image."}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "coco.yaml"
    manifest = prepare_coco(
        source,
        root,
        manifest_output=manifest_path,
        source="local fixture",
        license_name="test only",
    )
    assert manifest["splits"] == {"train": 0, "validation": 0, "test": 1}
    assert verify_dataset(root, load_dataset_manifest(manifest_path)) == []
    row = json.loads((root / "annotations/test.jsonl").read_text(encoding="utf-8"))
    assert row["image"] == "val2014/COCO_val2014_000000000001.jpg"
