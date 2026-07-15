"""Deterministic execution helpers and machine-readable environment capture."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from recoalign.schema_validation import validate_payload


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch when available."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False


def get_git_metadata(cwd: str | Path | None = None) -> dict[str, Any]:
    """Capture commit, branch, dirty state, and a digest of uncommitted changes."""
    root = Path(cwd or Path.cwd()).resolve()
    commit = _run_text(["git", "rev-parse", "HEAD"], cwd=root)
    if commit is None:
        return {
            "commit": None,
            "branch": None,
            "dirty": None,
            "untracked_count": 0,
            "diff_sha256": None,
        }

    branch = _run_text(["git", "branch", "--show-current"], cwd=root) or None
    status = _run_text(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=root
    )
    dirty = bool(status)
    untracked_bytes = _run_bytes(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root
    ) or b""
    untracked = [item for item in untracked_bytes.split(b"\0") if item]

    diff_sha256 = None
    if dirty:
        digest = hashlib.sha256()
        digest.update(_run_bytes(["git", "diff", "--binary", "HEAD"], cwd=root) or b"")
        for encoded_path in sorted(untracked):
            relative = Path(os.fsdecode(encoded_path))
            digest.update(encoded_path)
            path = root / relative
            if path.is_file():
                digest.update(_sha256_bytes(path))
        diff_sha256 = digest.hexdigest()

    return {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
        "untracked_count": len(untracked),
        "diff_sha256": diff_sha256,
    }


def collect_environment(cwd: str | Path | None = None) -> dict[str, Any]:
    """Collect package, Git, and accelerator metadata without requiring PyTorch."""
    freeze = _pip_freeze()
    freeze_digest = hashlib.sha256("\n".join(freeze).encode("utf-8")).hexdigest()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "captured_at": utc_now(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git": get_git_metadata(cwd),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "packages": {},
        "pip_freeze": freeze,
        "pip_freeze_sha256": freeze_digest,
    }

    for package in (
        "numpy",
        "Pillow",
        "PyYAML",
        "jsonschema",
        "torch",
        "torchvision",
        "open_clip_torch",
    ):
        try:
            payload["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            payload["packages"][package] = None

    accelerator: dict[str, Any] = {
        "cuda_available": False,
        "torch_cuda_version": None,
        "cudnn_version": None,
        "nvidia_driver_version": _nvidia_driver_version(),
        "devices": [],
    }
    try:
        import torch
    except ImportError:
        payload["accelerator"] = accelerator
        validate_payload("environment", payload)
        return payload

    accelerator["cuda_available"] = torch.cuda.is_available()
    accelerator["torch_cuda_version"] = torch.version.cuda
    accelerator["cudnn_version"] = torch.backends.cudnn.version()
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            accelerator["devices"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": [properties.major, properties.minor],
                }
            )
    payload["accelerator"] = accelerator
    validate_payload("environment", payload)
    return payload


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Write JSON atomically to avoid incomplete experiment records."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        temporary_path = Path(handle.name)
    temporary_path.replace(destination)


def _pip_freeze() -> list[str]:
    output = _run_text([sys.executable, "-m", "pip", "freeze", "--all"], cwd=Path.cwd())
    return sorted(line.strip() for line in (output or "").splitlines() if line.strip())


def _nvidia_driver_version() -> str | None:
    output = _run_text(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        cwd=Path.cwd(),
    )
    if not output:
        return None
    return output.splitlines()[0].strip() or None


def _run_text(args: list[str], *, cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _run_bytes(args: list[str], *, cwd: Path) -> bytes | None:
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            check=True,
            capture_output=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return completed.stdout


def _sha256_bytes(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.digest()
