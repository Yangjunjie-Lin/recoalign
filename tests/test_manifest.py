import json
from pathlib import Path
from typing import Any

import pytest

from recoalign.data.manifest import sha256_file, verify_manifest_files


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
        f"invalid image inventory manifest {missing_field}" in failure
        for failure in failures
    )
