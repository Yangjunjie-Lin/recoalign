"""Manifest loading, hashing, snapshotting, and local artifact verification."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
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


def verify_manifest_files(root: str | Path, manifest: dict[str, Any]) -> list[str]:
    """Return deterministic verification failures for files declared by a manifest."""
    failures: list[str] = []
    root_path = Path(root)
    for entry in manifest.get("files", []):
        failures.extend(_verify_file_entry(root_path, entry))

    processing = manifest.get("processing")
    inventory = processing.get("image_inventory") if isinstance(processing, dict) else None
    if inventory:
        failures.extend(_verify_inventory(root_path, inventory))
    return failures


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
    relative_path = Path(entry["path"])
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return [f"unsafe path: {relative_path}"]
    path = root / relative_path
    if not path.is_file():
        return [f"missing file: {relative_path}"]
    failures: list[str] = []
    expected_bytes = entry.get("bytes")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        failures.append(f"size mismatch: {relative_path}")
    expected_sha = entry.get("sha256")
    if expected_sha and sha256_file(path) != expected_sha:
        failures.append(f"sha256 mismatch: {relative_path}")
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
