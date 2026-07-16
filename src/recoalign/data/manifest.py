"""Manifest loading, hashing, snapshotting, and local artifact verification."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from recoalign.schema_validation import validate_payload


class ManifestError(ValueError):
    """Raised when a dataset or checkpoint manifest is invalid."""


def load_dataset_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a dataset manifest."""
    return _load_manifest(path, "dataset_manifest")


def load_checkpoint_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a checkpoint manifest."""
    return _load_manifest(path, "checkpoint_manifest")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a SHA-256 digest without loading a complete file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest_files(
    root: str | Path,
    manifest: dict[str, Any],
    *,
    require_hashed_inventory: bool = False,
) -> list[str]:
    """Return deterministic verification failures for files declared by a manifest."""
    failures: list[str] = []
    root_path = Path(root)
    for entry in manifest.get("files", []):
        failures.extend(_verify_file_entry(root_path, entry))

    processing = manifest.get("processing")
    inventory = processing.get("image_inventory") if isinstance(processing, dict) else None
    if require_hashed_inventory:
        for failure in verify_hashed_image_inventory(root_path, manifest):
            if failure not in failures:
                failures.append(failure)
    elif inventory:
        failures.extend(_verify_inventory(root_path, inventory))
    return failures


def verify_hashed_image_inventory(
    root: str | Path,
    manifest: dict[str, Any],
) -> list[str]:
    """Verify the complete, manifest-declared hashed image inventory contract."""
    return _verify_hashed_inventory(Path(root), manifest)


def verify_dataset(root: str | Path, manifest: dict[str, Any]) -> list[str]:
    """Backward-compatible dataset verification alias."""
    validate_payload("dataset_manifest", manifest)
    return verify_manifest_files(root, manifest)


def snapshot_manifest(source: str | Path, destination: str | Path) -> None:
    """Copy a committed manifest into a run directory without rewriting it."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _verify_file_entry(root: Path, entry: dict[str, Any]) -> list[str]:
    path_text = entry["path"]
    relative_path = Path(path_text)
    if not _is_safe_relative_path(path_text):
        return [f"unsafe path: {path_text}"]
    path = root / relative_path
    if not path.is_file():
        return [f"missing file: {path_text}"]
    failures: list[str] = []
    expected_bytes = entry.get("bytes")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        failures.append(f"size mismatch: {path_text}")
    expected_sha = entry.get("sha256")
    if expected_sha and sha256_file(path) != expected_sha:
        failures.append(f"sha256 mismatch: {path_text}")
    return failures


def _verify_inventory(root: Path, inventory: str) -> list[str]:
    relative_inventory = Path(inventory)
    if relative_inventory.is_absolute() or ".." in relative_inventory.parts:
        return [f"unsafe inventory path: {relative_inventory}"]
    inventory_path = root / relative_inventory
    if not inventory_path.is_file():
        return [f"missing image inventory: {relative_inventory}"]

    failures: list[str] = []
    seen: set[str] = set()
    with inventory_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(
                    f"invalid image inventory JSON at {relative_inventory}:{line_number}: {exc.msg}"
                )
                continue
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                failures.append(f"invalid image inventory row: {relative_inventory}:{line_number}")
                continue
            path_text = entry["path"]
            if path_text in seen:
                failures.append(f"duplicate image inventory path: {path_text}")
                continue
            seen.add(path_text)
            failures.extend(_verify_file_entry(root, entry))
    if not seen:
        failures.append(f"empty image inventory: {relative_inventory}")
    return failures


def _verify_hashed_inventory(root: Path, manifest: dict[str, Any]) -> list[str]:
    processing = manifest.get("processing")
    inventory = processing.get("image_inventory") if isinstance(processing, dict) else None
    if not isinstance(inventory, str) or not inventory.strip():
        return ["reportable image benchmarks require an image inventory"]
    if processing.get("image_hashes") is not True:
        return ["reportable image benchmarks require image_hashes=true"]
    if not _is_safe_relative_path(inventory):
        return [f"unsafe image inventory path: {inventory}"]

    failures: list[str] = []
    inventory_entries = [
        entry
        for entry in manifest.get("files", [])
        if isinstance(entry, dict) and entry.get("path") == inventory
    ]
    if not inventory_entries:
        failures.append(f"image inventory is not declared in manifest files: {inventory}")
    else:
        inventory_entry = inventory_entries[0]
        if not _is_nonnegative_integer(inventory_entry.get("bytes")):
            failures.append(f"invalid image inventory manifest bytes: {inventory}")
        if not _is_sha256(inventory_entry.get("sha256")):
            failures.append(f"invalid image inventory manifest sha256: {inventory}")
        failures.extend(_verify_file_entry(root, inventory_entry))

    inventory_path = root / Path(inventory)
    if not inventory_path.is_file():
        failures.append(f"missing image inventory: {inventory}")
        return failures

    seen: set[str] = set()
    nonempty_rows = 0
    with inventory_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            nonempty_rows += 1
            location = f"{inventory}:{line_number}"
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(f"invalid image inventory JSON at {location}: {exc.msg}")
                continue
            if not isinstance(entry, dict):
                failures.append(f"invalid image inventory row: {location}")
                continue

            path_text = entry.get("path")
            valid_path = isinstance(path_text, str) and bool(path_text.strip())
            if "path" not in entry:
                failures.append(f"image inventory row missing path: {location}")
            elif not valid_path:
                failures.append(f"invalid image inventory path: {location}")
            elif not _is_safe_relative_path(path_text):
                failures.append(f"unsafe image inventory path: {path_text}")
                valid_path = False
            elif path_text in seen:
                failures.append(f"duplicate image inventory path: {path_text}")
                valid_path = False
            else:
                seen.add(path_text)

            valid_bytes = _is_nonnegative_integer(entry.get("bytes"))
            if "bytes" not in entry:
                failures.append(f"image inventory row missing bytes: {location}")
            elif not valid_bytes:
                failures.append(f"invalid image inventory bytes: {location}")

            valid_sha = _is_sha256(entry.get("sha256"))
            if "sha256" not in entry:
                failures.append(f"image inventory row missing sha256: {location}")
            elif not valid_sha:
                failures.append(f"invalid image inventory sha256: {location}")

            if not (valid_path and valid_bytes and valid_sha):
                continue
            image_path = root / Path(path_text)
            if not image_path.is_file():
                failures.append(f"missing file: {path_text}")
                continue
            if image_path.stat().st_size != entry["bytes"]:
                failures.append(f"size mismatch: {path_text}")
            if sha256_file(image_path).lower() != entry["sha256"].lower():
                failures.append(f"sha256 mismatch: {path_text}")

    if nonempty_rows == 0:
        failures.append(f"empty image inventory: {inventory}")
    return failures


def _is_safe_relative_path(value: str) -> bool:
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    return not (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in posix_path.parts
        or ".." in windows_path.parts
    )


def _is_nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def _load_manifest(path: str | Path, schema_name: str) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ManifestError("manifest root must be a mapping")
    validate_payload(schema_name, payload)
    return payload
