from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from PIL import Image

from recoalign.data.compositional_preparation import prepare_winoground

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_winoground.py"


def _load_exporter() -> ModuleType:
    spec = importlib.util.spec_from_file_location("export_winoground", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXPORTER = _load_exporter()


def _row(official_id: int = 0) -> dict[str, Any]:
    return {
        "id": official_id,
        "image_0": Image.new("RGB", (3, 2), color=(255, 0, 0)),
        "image_1": Image.new("RGB", (3, 2), color=(0, 0, 255)),
        "caption_0": "A Red cup, left.",
        "caption_1": "left. cup, Red A",
        "tag": "Relation",
        "secondary_tag": ["Object-A", "Object-B"],
        "collapsed_tag": "Spatial",
    }


def _read_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "incoming" / "winoground.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_export_preserves_order_text_tags_and_metadata(tmp_path: Path) -> None:
    rows = [_row(7), _row(12)]

    summary = EXPORTER.export_winoground(rows, tmp_path, expected_samples=2)

    exported = _read_rows(tmp_path)
    assert summary["samples"] == 2
    assert summary["unique_sample_ids"] == 2
    assert summary["token_multiset_method"] == (
        "casefolded_alphanumeric_character_multiset_v1"
    )
    assert exported[0]["sample_id"] == "winoground-000007"
    assert exported[0]["caption_0"] == rows[0]["caption_0"]
    assert exported[0]["caption_1"] == rows[0]["caption_1"]
    assert exported[0]["image_0"] == "000007_image_0.png"
    assert exported[0]["image_1"] == "000007_image_1.png"
    assert exported[0]["tags"] == ["Relation", "Object-A", "Object-B", "Spatial"]
    assert exported[0]["metadata"] == {
        "official_id": 7,
        "official_tag_fields": {
            "collapsed_tag": "Spatial",
            "secondary_tag": ["Object-A", "Object-B"],
            "tag": "Relation",
        },
        "source_dataset": "facebook/winoground",
        "source_split": "test",
    }
    with Image.open(tmp_path / "images" / exported[0]["image_0"]) as image_0:
        assert image_0.getpixel((0, 0)) == (255, 0, 0)
    with Image.open(tmp_path / "images" / exported[0]["image_1"]) as image_1:
        assert image_1.getpixel((0, 0)) == (0, 0, 255)


def test_export_preserves_empty_official_secondary_tag(tmp_path: Path) -> None:
    row = _row()
    row["secondary_tag"] = ""

    EXPORTER.export_winoground([row], tmp_path, expected_samples=1)

    exported = _read_rows(tmp_path)[0]
    assert exported["tags"] == ["Relation", "Spatial"]
    assert exported["metadata"]["official_tag_fields"]["secondary_tag"] == ""


def test_export_dry_run_validates_without_writing(tmp_path: Path) -> None:
    summary = EXPORTER.export_winoground(
        [_row()], tmp_path / "missing", dry_run=True, max_samples=1
    )

    assert summary["dry_run"] is True
    assert summary["formal_export"] is False
    assert not (tmp_path / "missing").exists()


def test_export_refuses_existing_outputs_without_overwrite(tmp_path: Path) -> None:
    EXPORTER.export_winoground([_row()], tmp_path, expected_samples=1)

    with pytest.raises(FileExistsError, match="--overwrite"):
        EXPORTER.export_winoground([_row()], tmp_path, expected_samples=1)

    replacement = _row()
    replacement["caption_0"] = "replacement"
    EXPORTER.export_winoground([replacement], tmp_path, expected_samples=1, overwrite=True)
    assert _read_rows(tmp_path)[0]["caption_0"] == "replacement"


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda rows: rows.append(_row(0)), "duplicate official id"),
        (lambda rows: rows[0].update(caption_0=""), "caption_0 must be"),
        (lambda rows: rows[0].pop("image_1"), "missing field 'image_1'"),
        (lambda rows: rows[0].update(id="../unsafe"), "non-negative integer"),
    ],
)
def test_export_rejects_invalid_rows(tmp_path: Path, mutation, error: str) -> None:
    rows = [_row()]
    mutation(rows)
    output_root = tmp_path / "output"

    with pytest.raises(ValueError, match=error):
        EXPORTER.export_winoground(rows, output_root, dry_run=True, max_samples=len(rows))
    assert not output_root.exists()


def test_export_requires_400_samples_for_formal_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires 400 samples"):
        EXPORTER.export_winoground([_row()], tmp_path)


def test_export_output_is_accepted_by_prepare_winoground(tmp_path: Path) -> None:
    dataset_root = tmp_path / "winoground"
    EXPORTER.export_winoground([_row()], dataset_root, expected_samples=1)

    manifest = prepare_winoground(
        dataset_root / "incoming" / "winoground.jsonl",
        dataset_root,
        manifest_output=tmp_path / "winoground.yaml",
        source="synthetic test fixture",
        license_name="test only",
        hash_images=True,
    )

    assert manifest["splits"] == {"test": 1}
    assert manifest["processing"]["image_hashes"] is True
    assert manifest["processing"]["caption_token_multiset_match_rate"] == 100.0
    assert len((dataset_root / "annotations" / "test.jsonl").read_text().splitlines()) == 1
