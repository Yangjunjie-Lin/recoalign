#!/usr/bin/env python3
"""Diagnose the local ReCoAlign Python, filesystem, PyTorch, and OpenCLIP setup."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path


def check_writable(path: Path) -> tuple[bool, str]:
    """Attempt a small write without leaving a file behind."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, delete=True):
            pass
        return True, "writable"
    except OSError as exc:
        return False, f"not writable: {exc}"


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Executable: {sys.executable}")
    print(f"Conda environment: {os.environ.get('CONDA_DEFAULT_ENV', 'not detected')}")

    for path in (Path.home(), Path("/tmp")):
        ok, message = check_writable(path)
        print(f"Filesystem {path}: {message}")
        if not ok:
            return 1

    disk = shutil.disk_usage(Path.home())
    print(f"Home free disk: {disk.free / 1024**3:.1f} GiB")

    if importlib.util.find_spec("torch") is None:
        print("PyTorch: not installed")
        return 1

    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime in PyTorch: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        print(f"GPU: {properties.name}")
        print(f"VRAM: {properties.total_memory / 1024**3:.1f} GiB")

    if importlib.util.find_spec("open_clip") is None:
        print("OpenCLIP: not installed")
        return 1

    import open_clip

    print(f"OpenCLIP: {getattr(open_clip, '__version__', 'installed')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
