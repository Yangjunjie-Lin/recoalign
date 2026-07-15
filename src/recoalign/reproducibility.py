"""Utilities for deterministic runs and machine-readable environment capture."""

from __future__ import annotations

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


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int) -> None:
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


def get_git_commit(cwd: str | Path | None = None) -> str | None:
    """Return the current Git commit, or ``None`` outside a repository."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def collect_environment(cwd: str | Path | None = None) -> dict[str, Any]:
    """Collect versions and accelerator metadata without requiring PyTorch."""
    payload: dict[str, Any] = {
        "schema_version": 1,
        "captured_at": utc_now(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": get_git_commit(cwd),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "packages": {},
    }

    for package in ("numpy", "Pillow", "PyYAML", "torch", "torchvision", "open_clip_torch"):
        try:
            payload["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            payload["packages"][package] = None

    try:
        import torch
    except ImportError:
        payload["accelerator"] = {"cuda_available": False, "devices": []}
        return payload

    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": [properties.major, properties.minor],
                }
            )
    payload["accelerator"] = {
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "devices": devices,
    }
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
