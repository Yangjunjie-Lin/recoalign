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
from typing import TextIO


def safe_print(message: object, *, stream: TextIO | None = None) -> None:
    """Print diagnostic text even when the console cannot encode every character."""
    output = stream if stream is not None else sys.stdout
    encoding = getattr(output, "encoding", None) or "utf-8"
    text = str(message).encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(text, file=output)


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
    safe_print(f"Python: {sys.version.split()[0]}")
    safe_print(f"Platform: {platform.platform()}")
    safe_print(f"Executable: {sys.executable}")
    safe_print(f"Conda environment: {os.environ.get('CONDA_DEFAULT_ENV', 'not detected')}")

    for path in (Path.home(), Path("/tmp")):
        ok, message = check_writable(path)
        safe_print(f"Filesystem {path}: {message}")
        if not ok:
            return 1

    disk = shutil.disk_usage(Path.home())
    safe_print(f"Home free disk: {disk.free / 1024**3:.1f} GiB")

    if importlib.util.find_spec("torch") is None:
        safe_print("PyTorch: not installed")
        return 1

    import torch

    safe_print(f"PyTorch: {torch.__version__}")
    safe_print(f"CUDA runtime in PyTorch: {torch.version.cuda}")
    safe_print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        safe_print(f"GPU: {properties.name}")
        safe_print(f"VRAM: {properties.total_memory / 1024**3:.1f} GiB")

    if importlib.util.find_spec("open_clip") is None:
        safe_print("OpenCLIP: not installed")
        return 1

    import open_clip

    safe_print(f"OpenCLIP: {getattr(open_clip, '__version__', 'installed')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
