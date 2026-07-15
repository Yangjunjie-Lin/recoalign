"""Dataset manifest loading and optional local file verification."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


class DatasetManifestError(ValueError):
    """Raised when a dataset manifest violates the Phase-0 contract."""


def load_dataset_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a dataset manifest."""
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise DatasetManifestError("manifest root must be a mapping")

    for field in ("schema_version", "name", "version", "source", "license", "splits"):
        if field not in payload:
            raise DatasetManifestError(f"missing dataset manifest field: {field}")
    if not isinstance(payload["splits"], dict) or not payload["splits"]:
        raise DatasetManifestError("splits must be a non-empty mapping")
    files = payload.get("files", [])
    if not isinstance(files, list):
        raise DatasetManifestError("files must be a list")
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise DatasetManifestError("every file entry must contain a string path")
    return payload


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset(root: str | Path, manifest: dict[str, Any]) -> list[str]:
    """Return human-readable verification failures for declared files."""
    failures: list[str] = []
    root_path = Path(root)
    for entry in manifest.get("files", []):
        relative_path = Path(entry["path"])
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
