"""Manifest loading, hashing, snapshotting, and local artifact verification."""

from __future__ import annotations

import hashlib
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
        relative_path = Path(entry["path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            failures.append(f"unsafe path: {relative_path}")
            continue
        path = root_path / relative_path
        if not path.is_file():
            failures.append(f"missing file: {relative_path}")
            continue
        expected_bytes = entry.get("bytes")
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            failures.append(f"size mismatch: {relative_path}")
        expected_sha = entry.get("sha256")
        if expected_sha and sha256_file(path) != expected_sha:
            failures.append(f"sha256 mismatch: {relative_path}")
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
