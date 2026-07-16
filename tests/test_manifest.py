import json
from pathlib import Path
from typing import Any

import pytest

from recoalign.data.manifest import (
    sha256_file,
    verify_annotation_inventory_coverage,
    verify_manifest_files,
)


def _hashed_inventory_fixture(
    tmp_path: Path,
    *,
    rows: list[dict[str, Any]] | None = None,
) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "dataset"
    image_path = root / "images" / "synthetic-image.bin"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"synthetic-image-bytes")

    inventory_path = root / "inventories" / "images.jsonl"
    inventory_path.parent.mkdir(parents=True)
    default_row = {
        "path": "images/synthetic-image.bin",
        "bytes": image_path.stat().st_size,
        "sha256": sha256_file(image_path),
    }
    inventory_path.write_text(
        "".join(json.dumps(row) + "\n" for row in (rows or [default_row])),
        encoding="utf-8",
    )
    manifest = {
        "files": [
            {
                "path": "inventories/images.jsonl",
                "bytes": inventory_path.stat().st_size,
                "sha256": sha256_file(inventory_path),
            }
        ],
        "processing": {
            "image_inventory": "inventories/images.jsonl",
            "image_hashes": True,
        },
    }
    return root, manifest


def _valid_row(root: Path) -> dict[str, Any]:
    image_path = root / "images" / "synthetic-image.bin"
    return {
        "path": "images/synthetic-image.bin",
        "bytes": image_path.stat().st_size,
        "sha256": sha256_file(image_path),
    }


def _rewrite_inventory(
    root: Path,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    inventory_path = root / "inventories" / "images.jsonl"
    inventory_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest["files"][0]["bytes"] = inventory_path.stat().st_size
    manifest["files"][0]["sha256"] = sha256_file(inventory_path)


def test_complete_hashed_image_inventory_passes(tmp_path: Path) -> None:
    root, manifest = _hashed_inventory_fixture(tmp_path)

    assert verify_manifest_files(root, manifest, require_hashed_inventory=True) == []


def test_hashed_image_inventory_requires_declared_path(tmp_path: Path) -> None:
    root, manifest = _hashed_inventory_fixture(tmp_path)
    manifest["processing"].pop("image_inventory")

    failures = verify_manifest_files(root, manifest, require_hashed_inventory=True)

    assert failures == ["reportable image benchmarks require an image inventory"]


def test_image_inventory_must_be_manifest_declared(tmp_path: Path) -> None:
    root, manifest = _hashed_inventory_fixture(tmp_path)
    manifest["files"] = []

    failures = verify_manifest_files(root, manifest, require_hashed_inventory=True)

    assert "image inventory is not declared in manifest files" in failures[0]


def test_declared_image_inventory_file_must_exist(tmp_path: Path) -> None:
    root, manifest = _hashed_inventory_fixture(tmp_path)
    (root / "inventories" / "images.jsonl").unlink()

    failures = verify_manifest_files(root, manifest, require_hashed_inventory=True)

    assert "missing image inventory: inventories/images.jsonl" in failures


def test_image_inventory_row_requires_path(tmp_path: Path) -> None:
    root, manifest = _hashed_inventory_fixture(tmp_path)
    row = _valid_row(root)
    row.pop("path")
    _rewrite_inventory(root, manifest, [row])

    failures = verify_manifest_files(root, manifest, require_hashed_inventory=True)

    assert "image inventory row missing path: inventories/images.jsonl:1" in failures


@pytest.mark.parametrize("missing_field", ["bytes", "sha256"])
def test_image_inventory_row_requires_hash_metadata(
    tmp_path: Path,
    missing_field: str,
) -> None:
    root, manifest = _hashed_inventory_fixture(tmp_path)
    row = _valid_row(root)
    row.pop(missing_field)
    _rewrite_inventory(root, manifest, [row])

    failures = verify_manifest_files(root, manifest, require_hashed_inventory=True)

    assert f"image inventory row missing {missing_field}" in failures[-1]


@pytest.mark.parametrize("sha256", ["abc", "a" * 63, "a" * 65, "g" * 64])
def test_image_inventory_row_rejects_invalid_sha256(tmp_path: Path, sha256: str) -> None:
    root, manifest = _hashed_inventory_fixture(tmp_path)
    row = _valid_row(root)
    row["sha256"] = sha256
    _rewrite_inventory(root, manifest, [row])

    failures = verify_manifest_files(root, manifest, require_hashed_inventory=True)

    assert "invalid image inventory sha256" in failures[-1]


@pytest.mark.parametrize("byte_count", ["123", True, -1])
def test_image_inventory_row_rejects_invalid_bytes(
    tmp_path: Path,
    byte_count: object,
) -> None:
    root, manifest = _hashed_inventory_fixture(tmp_path)
    row = _valid_row(root)
    row["bytes"] = byte_count
    _rewrite_inventory(root, manifest, [row])

    failures = verify_manifest_files(root, manifest, require_hashed_inventory=True)

    assert "invalid image inventory bytes" in failures[-1]


def test_image_inventory_rejects_image_hash_mismatch(tmp_path: Path) -> None:
    root, manifest = _hashed_inventory_fixture(tmp_path)
    row = _valid_row(root)
    row["sha256"] = "0" * 64
    _rewrite_inventory(root, manifest, [row])

    failures = verify_manifest_files(root, manifest, require_hashed_inventory=True)

    assert "sha256 mismatch: images/synthetic-image.bin" in failures


def test_image_inventory_rejects_image_size_mismatch(tmp_path: Path) -> None:
    root, manifest = _hashed_inventory_fixture(tmp_path)
    row = _valid_row(root)
    row["bytes"] += 1
    _rewrite_inventory(root, manifest, [row])

    failures = verify_manifest_files(root, manifest, require_hashed_inventory=True)

    assert "size mismatch: images/synthetic-image.bin" in failures


def test_image_inventory_must_not_be_empty(tmp_path: Path) -> None:
    root, manifest = _hashed_inventory_fixture(tmp_path)
    _rewrite_inventory(root, manifest, [])

    failures = verify_manifest_files(root, manifest, require_hashed_inventory=True)

    assert "empty image inventory: inventories/images.jsonl" in failures


def test_image_inventory_rejects_duplicate_image_path(tmp_path: Path) -> None:
    root, manifest = _hashed_inventory_fixture(tmp_path)
    row = _valid_row(root)
    _rewrite_inventory(root, manifest, [row, row])

    failures = verify_manifest_files(root, manifest, require_hashed_inventory=True)

    assert "duplicate image inventory path: images/synthetic-image.bin" in failures


@pytest.mark.parametrize(
    "unsafe_path",
    ["/tmp/image.png", "../image.png", "images/../../secret", "C:\\images\\a.png"],
)
def test_image_inventory_rejects_unsafe_image_path(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    root, manifest = _hashed_inventory_fixture(tmp_path)
    row = _valid_row(root)
    row["path"] = unsafe_path
    _rewrite_inventory(root, manifest, [row])

    failures = verify_manifest_files(root, manifest, require_hashed_inventory=True)

    assert f"unsafe image inventory path: {unsafe_path}" in failures


@pytest.mark.parametrize("missing_field", ["bytes", "sha256"])
def test_inventory_manifest_entry_requires_hash_metadata(
    tmp_path: Path,
    missing_field: str,
) -> None:
    root, manifest = _hashed_inventory_fixture(tmp_path)
    manifest["files"][0].pop(missing_field)

    failures = verify_manifest_files(root, manifest, require_hashed_inventory=True)

    assert any(
        f"invalid image inventory manifest {missing_field}" in failure for failure in failures
    )


def _annotation_coverage_fixture(tmp_path: Path) -> tuple[Path, dict[str, Any], str]:
    root = tmp_path / "coverage-dataset"
    images = root / "images"
    annotations = root / "annotations"
    inventories = root / "inventories"
    images.mkdir(parents=True)
    annotations.mkdir()
    inventories.mkdir()
    for name, content in (("a.png", b"synthetic-a"), ("b.png", b"synthetic-b")):
        (images / name).write_bytes(content)
    inventory_path = inventories / "images.jsonl"
    inventory_rows = [
        {
            "path": f"images/{name}",
            "bytes": (images / name).stat().st_size,
            "sha256": sha256_file(images / name),
        }
        for name in ("a.png", "b.png")
    ]
    inventory_path.write_text(
        "".join(json.dumps(row) + "\n" for row in inventory_rows),
        encoding="utf-8",
    )
    annotation_path = annotations / "test.jsonl"
    annotation_path.write_text(
        json.dumps(
            {
                "sample_id": "sample-0",
                "image_0": "a.png",
                "image_1": "b.png",
                "caption_0": "synthetic caption",
                "caption_1": "caption synthetic",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    annotation_sha = sha256_file(annotation_path)
    manifest = {
        "splits": {"test": 1},
        "files": [
            {
                "path": "annotations/test.jsonl",
                "bytes": annotation_path.stat().st_size,
                "sha256": annotation_sha,
            },
            {
                "path": "inventories/images.jsonl",
                "bytes": inventory_path.stat().st_size,
                "sha256": sha256_file(inventory_path),
            },
        ],
        "processing": {
            "format": "recoalign-paired-matrix-jsonl-v1",
            "image_inventory": "inventories/images.jsonl",
            "image_hashes": True,
            "caption_alphanumeric_character_multiset_match_rate": 100.0,
            "caption_alphanumeric_character_multiset_method": (
                "casefolded_alphanumeric_character_multiset_v1"
            ),
        },
    }
    return root, manifest, annotation_sha


def _refresh_annotation_entry(root: Path, manifest: dict[str, Any]) -> str:
    path = root / "annotations" / "test.jsonl"
    entry = next(item for item in manifest["files"] if item["path"] == "annotations/test.jsonl")
    entry["bytes"] = path.stat().st_size
    entry["sha256"] = sha256_file(path)
    return entry["sha256"]


def _refresh_coverage_inventory(root: Path, manifest: dict[str, Any]) -> None:
    path = root / "inventories" / "images.jsonl"
    entry = next(item for item in manifest["files"] if item["path"] == "inventories/images.jsonl")
    entry["bytes"] = path.stat().st_size
    entry["sha256"] = sha256_file(path)


def test_annotation_inventory_exact_coverage_passes(tmp_path: Path) -> None:
    root, manifest, annotation_sha = _annotation_coverage_fixture(tmp_path)

    assert (
        verify_annotation_inventory_coverage(
            root,
            manifest,
            split="test",
            expected_annotation_sha256=annotation_sha,
        )
        == []
    )


def test_annotation_inventory_rejects_missing_referenced_image(tmp_path: Path) -> None:
    root, manifest, annotation_sha = _annotation_coverage_fixture(tmp_path)
    path = root / "inventories" / "images.jsonl"
    rows = path.read_text().splitlines()
    path.write_text(rows[0] + "\n", encoding="utf-8")
    _refresh_coverage_inventory(root, manifest)

    failures = verify_annotation_inventory_coverage(
        root, manifest, split="test", expected_annotation_sha256=annotation_sha
    )

    assert "missing 1 annotation-referenced images" in failures[0]


def test_annotation_inventory_rejects_unreferenced_extra_image(tmp_path: Path) -> None:
    root, manifest, _ = _annotation_coverage_fixture(tmp_path)
    path = root / "annotations" / "test.jsonl"
    row = json.loads(path.read_text())
    row["image_1"] = "a.png"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    annotation_sha = _refresh_annotation_entry(root, manifest)

    failures = verify_annotation_inventory_coverage(
        root, manifest, split="test", expected_annotation_sha256=annotation_sha
    )

    assert "contains 1 images not referenced" in failures[0]


def test_annotation_coverage_requires_manifest_declaration(tmp_path: Path) -> None:
    root, manifest, annotation_sha = _annotation_coverage_fixture(tmp_path)
    manifest["files"] = [
        entry for entry in manifest["files"] if entry["path"] != "annotations/test.jsonl"
    ]

    failures = verify_annotation_inventory_coverage(
        root, manifest, split="test", expected_annotation_sha256=annotation_sha
    )

    assert "not declared in manifest files" in failures[0]


@pytest.mark.parametrize("field", ["bytes", "sha256"])
def test_annotation_coverage_requires_manifest_hash_metadata(tmp_path: Path, field: str) -> None:
    root, manifest, annotation_sha = _annotation_coverage_fixture(tmp_path)
    manifest["files"][0].pop(field)

    failures = verify_annotation_inventory_coverage(
        root, manifest, split="test", expected_annotation_sha256=annotation_sha
    )

    assert f"invalid evaluation annotation manifest {field}" in failures[0]


def test_annotation_coverage_rejects_file_hash_mismatch(tmp_path: Path) -> None:
    root, manifest, annotation_sha = _annotation_coverage_fixture(tmp_path)
    (root / "annotations" / "test.jsonl").write_text("{}\n", encoding="utf-8")

    failures = verify_annotation_inventory_coverage(
        root, manifest, split="test", expected_annotation_sha256=annotation_sha
    )

    assert any("sha256 mismatch" in failure for failure in failures)


def test_annotation_coverage_rejects_evaluation_digest_mismatch(tmp_path: Path) -> None:
    root, manifest, _ = _annotation_coverage_fixture(tmp_path)

    failures = verify_annotation_inventory_coverage(
        root, manifest, split="test", expected_annotation_sha256="0" * 64
    )

    assert "evaluation annotation sha256 does not match dataset manifest" in failures


def test_annotation_coverage_rejects_duplicate_sample_id(tmp_path: Path) -> None:
    root, manifest, _ = _annotation_coverage_fixture(tmp_path)
    path = root / "annotations" / "test.jsonl"
    path.write_text(path.read_text() * 2, encoding="utf-8")
    annotation_sha = _refresh_annotation_entry(root, manifest)

    failures = verify_annotation_inventory_coverage(
        root, manifest, split="test", expected_annotation_sha256=annotation_sha
    )

    assert "duplicate annotation sample ID: sample-0" in failures


def test_annotation_coverage_rejects_unsafe_image_path(tmp_path: Path) -> None:
    root, manifest, _ = _annotation_coverage_fixture(tmp_path)
    path = root / "annotations" / "test.jsonl"
    row = json.loads(path.read_text())
    row["image_0"] = "../secret.png"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    annotation_sha = _refresh_annotation_entry(root, manifest)

    failures = verify_annotation_inventory_coverage(
        root, manifest, split="test", expected_annotation_sha256=annotation_sha
    )

    assert "unsafe annotation image path: ../secret.png" in failures
