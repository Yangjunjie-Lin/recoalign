"""Build deterministic ReCoAlign source and evidence tar.gz archives."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "recoalign-negative-evidence-v1"
PACKAGE_REL = "release/recoalign-negative-evidence-v1"
SOURCE_EXCLUDED_PREFIXES = (
    ".git/",
    ".local_artifacts/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".venv/",
    "checkpoints/",
    "data/raw/",
    "datasets/raw/",
    "outputs/",
    "runs/",
    f"{PACKAGE_REL}/archives/",
    f"{PACKAGE_REL}/checksums/SHA256SUMS",
)
EVIDENCE_PREFIXES = (
    ".zenodo.json",
    "CITATION.cff",
    "LICENSE",
    "codemeta.json",
    "docs/evidence_map.yaml",
    "reports/final_research_line_adjudication.md",
    "reports/paper_evidence/",
    "reports/paper_readiness_report.md",
    "reports/release_branch_integration.md",
    "reports/final_branch_and_tag_inventory.md",
    "research/current_line_closeout.yaml",
    "research/final_pivot_candidate_matrix.yaml",
    "research/final_pivot_literature_audit.md",
    "research/pivot_validation/",
    "research/causal_separation/PIVOT_EXP_A2/",
    "research/construct_validity/PIVOT_EXP_A3/",
    "research/construct_validity/PIVOT_EXP_A3R/",
    "research/construct_validity/PIVOT_EXP_A3P/",
    "release/FINAL_STATUS.yaml",
    f"{PACKAGE_REL}/",
    "release/validation/",
)


def _git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def _paths(ref: str) -> list[str]:
    raw = _git("ls-tree", "-r", "-z", "--name-only", ref)
    return sorted(path for path in raw.decode("utf-8").split("\0") if path)


def _included_source(path: str) -> bool:
    return not any(path == prefix or path.startswith(prefix) for prefix in SOURCE_EXCLUDED_PREFIXES)


def _included_evidence(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix) for prefix in EVIDENCE_PREFIXES)


def _blob(ref: str, path: str) -> bytes:
    return _git("show", f"{ref}:{path}")


def _archive(ref: str, paths: list[str], destination: Path) -> tuple[str, int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw_tar = io.BytesIO()
    with tarfile.open(fileobj=raw_tar, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in paths:
            payload = _blob(ref, path)
            info = tarfile.TarInfo(name=f"{PREFIX}/{path}")
            info.size = len(payload)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = 0o755 if path.startswith("scripts/") and path.endswith(".py") else 0o644
            archive.addfile(info, io.BytesIO(payload))
    raw_tar.seek(0)
    with destination.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
            compressed.write(raw_tar.getvalue())
    payload = destination.read_bytes()
    return hashlib.sha256(payload).hexdigest(), len(payload), len(paths)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--output-dir", type=Path, default=ROOT / PACKAGE_REL / "archives")
    args = parser.parse_args()
    ref = _git("rev-parse", "--verify", f"{args.ref}^{{commit}}").decode().strip()
    all_paths = _paths(ref)
    source_paths = [path for path in all_paths if _included_source(path)]
    evidence_paths = [path for path in all_paths if _included_evidence(path)]
    outputs = (
        (
            args.output_dir / f"{PREFIX}-source.tar.gz",
            source_paths,
        ),
        (
            args.output_dir / f"{PREFIX}-evidence.tar.gz",
            evidence_paths,
        ),
    )
    print(f"ref: {ref}")
    for destination, paths in outputs:
        digest, size, count = _archive(ref, paths, destination)
        print(f"{destination.name}: sha256={digest} bytes={size} files={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
