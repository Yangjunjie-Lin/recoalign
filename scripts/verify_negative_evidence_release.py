"""Integrity-only verifier for the ReCoAlign negative-evidence release."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "release" / "recoalign-negative-evidence-v1"
FROZEN_MANIFESTS = (
    ("reports/paper_evidence/artifact_manifest.yaml", "paper"),
    ("research/pivot_validation/results/artifact_manifest.yaml", "standard"),
    ("research/causal_separation/PIVOT_EXP_A2/results/artifact_manifest.yaml", "standard"),
    ("research/construct_validity/PIVOT_EXP_A3/v1_artifact_manifest.yaml", "standard"),
    ("research/construct_validity/PIVOT_EXP_A3R/final_artifact_manifest.yaml", "standard"),
    ("research/construct_validity/PIVOT_EXP_A3P/results/artifact_manifest.yaml", "standard"),
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping in {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_entries(path: Path, kind: str) -> list[tuple[str, Path, dict[str, Any]]]:
    manifest = _load_yaml(path)
    entries: list[tuple[str, Path, dict[str, Any]]] = []
    if kind == "paper":
        for study_id, study in manifest["experiments"].items():
            for role in ("metrics", "decision_report", "predictions"):
                metadata = study[role]
                entries.append((f"{study_id}:{role}", path.parent / metadata["path"], metadata))
        return entries
    for name, metadata in manifest["artifacts"].items():
        if isinstance(metadata, str):
            metadata = {"sha256": metadata}
        entries.append((name, path.parent / name, metadata))
    return entries


def _verify_frozen_manifests(failures: list[str]) -> dict[str, Any]:
    checked_entries = 0
    readable_gzip = 0
    for relative, kind in FROZEN_MANIFESTS:
        manifest_path = ROOT / relative
        for name, artifact, metadata in _manifest_entries(manifest_path, kind):
            checked_entries += 1
            label = f"{relative}:{name}"
            if not artifact.is_file():
                failures.append(f"missing frozen artifact: {label}")
                continue
            payload = artifact.read_bytes()
            if hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
                failures.append(f"frozen artifact sha256 mismatch: {label}")
            if "bytes" in metadata and len(payload) != metadata["bytes"]:
                failures.append(f"frozen artifact byte-count mismatch: {label}")
            if artifact.suffix == ".gz":
                try:
                    raw = gzip.decompress(payload)
                    readable_gzip += 1
                except (EOFError, OSError) as error:
                    failures.append(f"unreadable gzip artifact: {label}: {type(error).__name__}")
                    continue
                if (
                    "decompressed_sha256" in metadata
                    and hashlib.sha256(raw).hexdigest() != metadata["decompressed_sha256"]
                ):
                    failures.append(f"decompressed sha256 mismatch: {label}")
                if "decompressed_bytes" in metadata and len(raw) != metadata["decompressed_bytes"]:
                    failures.append(f"decompressed byte-count mismatch: {label}")
    return {"manifests": len(FROZEN_MANIFESTS), "entries": checked_entries, "gzip": readable_gzip}


def _verify_release_manifest(failures: list[str]) -> int:
    manifest = _load_yaml(PACKAGE / "RELEASE_MANIFEST.yaml")
    checked = 0
    for record in manifest["key_files"]:
        checked += 1
        expected = record["sha256"]
        path = ROOT / record["path"]
        if expected.startswith("pending_"):
            failures.append(f"unresolved release hash: {record['path']}")
        elif not path.is_file():
            failures.append(f"missing release key file: {record['path']}")
        elif _sha256(path) != expected:
            failures.append(f"release key-file sha256 mismatch: {record['path']}")
    return checked


def _verify_checksums(failures: list[str], require_archives: bool) -> int:
    checksum_path = PACKAGE / "checksums" / "SHA256SUMS"
    if not checksum_path.is_file():
        failures.append("missing checksums/SHA256SUMS")
        return 0
    checked = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", maxsplit=1)
        target = PACKAGE / relative
        if not target.is_file():
            if require_archives or not relative.startswith("archives/"):
                failures.append(f"missing checksummed file: {relative}")
            continue
        checked += 1
        if _sha256(target) != digest:
            failures.append(f"SHA256SUMS mismatch: {relative}")
    return checked


def verify(require_archives: bool = False) -> dict[str, Any]:
    failures: list[str] = []
    evidence = _load_yaml(ROOT / "docs" / "evidence_map.yaml")
    closeout = _load_yaml(ROOT / "research" / "current_line_closeout.yaml")
    summary = evidence["summary"]
    expected = {
        "program_decision": evidence.get("program_decision") == "TERMINATE_CURRENT_PROGRAM",
        "claim_count": summary.get("total_claims") == 14,
        "pending_claims": summary.get("pending") == 0,
        "selected_candidate": closeout.get("final_pivot_gate", {}).get("selected_candidate")
        == "STOP",
        "paper_writing_allowed": closeout.get("authorization", {}).get("paper_writing_allowed")
        is False,
        "archive_and_release_allowed": closeout.get("authorization", {}).get(
            "archive_and_release_allowed"
        )
        is True,
    }
    failures.extend(
        f"frozen state mismatch: {name}" for name, passed in expected.items() if not passed
    )
    frozen = _verify_frozen_manifests(failures)
    release_files = _verify_release_manifest(failures)
    checksummed_files = _verify_checksums(failures, require_archives)
    result = {
        "valid": not failures,
        "frozen_state": expected,
        "frozen_artifacts": frozen,
        "release_key_files": release_files,
        "checksummed_files_present": checksummed_files,
        "archives_required": require_archives,
        "failures": failures,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-archives", action="store_true")
    args = parser.parse_args()
    result = verify(require_archives=args.require_archives)
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
