"""Auditable model/optimizer/scheduler checkpoints with exact resume state."""

from __future__ import annotations

import hashlib
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from recoalign.reproducibility import get_git_metadata, utc_now


class TrainingCheckpointManager:
    def __init__(
        self,
        directory: str | Path,
        *,
        config: dict[str, Any],
        dataset_manifest: dict[str, Any] | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.dataset_manifest = dataset_manifest or {}
        self.project_root = Path(project_root or Path.cwd()).resolve()

    def save(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        epoch: int,
        global_step: int,
        metrics: dict[str, Any],
    ) -> Path:
        if epoch < 0 or global_step < 0:
            raise ValueError("checkpoint epoch and global_step must be non-negative")
        path = self.directory / f"epoch_{epoch:04d}.pt"
        payload = {
            "format_version": 1,
            "method": "recoalign",
            "saved_at": utc_now(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": self.config,
            "dataset_manifest": self.dataset_manifest,
            "metrics": metrics,
            "git": get_git_metadata(self.project_root),
            "rng_state": _rng_state(),
        }
        _atomic_torch_save(path, payload)
        digest = _sha256(path)
        manifest = {
            "schema_version": 1,
            "latest": path.name,
            "sha256": digest,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "contains": [
                "model weights",
                "optimizer state",
                "scheduler state",
                "resolved config",
                "git metadata",
                "dataset manifest",
                "training metrics",
                "rng state",
            ],
        }
        (self.directory.parent / "checkpoint_manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )
        return path

    def load(
        self,
        path: str | Path,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        map_location: str | torch.device = "cpu",
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"training checkpoint does not exist: {source}")
        try:
            payload = torch.load(source, map_location=map_location, weights_only=False)
        except TypeError:
            payload = torch.load(source, map_location=map_location)
        if not isinstance(payload, dict) or payload.get("method") != "recoalign":
            raise ValueError("checkpoint is not an auditable ReCoAlign training checkpoint")
        if payload.get("config") != self.config:
            raise ValueError("resume checkpoint resolved config differs from this training run")
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        if restore_rng:
            _restore_rng_state(payload.get("rng_state", {}))
        return payload


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["TrainingCheckpointManager"]
