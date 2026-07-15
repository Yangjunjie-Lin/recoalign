"""Inference-only OpenCLIP adapter used by reproducible baseline evaluation."""

from __future__ import annotations

import importlib.metadata
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class OpenCLIPConfig:
    """Configuration required to load an OpenCLIP checkpoint."""

    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    device: str = "cuda"
    precision: str = "amp"


class OpenCLIPEncoder:
    """Load OpenCLIP and expose normalized NumPy image/text embeddings."""

    def __init__(self, config: OpenCLIPConfig | None = None) -> None:
        self.config = config or OpenCLIPConfig()
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "OpenCLIP support is not installed. Install PyTorch first, then run "
                "`pip install -e '.[openclip]'`."
            ) from exc
        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        if self.config.precision not in {"fp32", "amp", "amp_bf16"}:
            raise ValueError("precision must be one of: fp32, amp, amp_bf16")

        self._torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.config.model_name,
            pretrained=self.config.pretrained,
            device=self.config.device,
        )
        self.tokenizer = open_clip.get_tokenizer(self.config.model_name)
        self.model.eval()
        if self.config.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.config.device)

    @property
    def fingerprint(self) -> dict[str, Any]:
        """Return model identity used in cache keys and evaluation metadata."""
        try:
            version = importlib.metadata.version("open_clip_torch")
        except importlib.metadata.PackageNotFoundError:
            version = None
        return {
            "framework": "open_clip",
            "framework_version": version,
            "model": self.config.model_name,
            "pretrained": self.config.pretrained,
            "device": self.config.device,
            "precision": self.config.precision,
        }

    @property
    def runtime_metadata(self) -> dict[str, Any]:
        """Return accelerator memory diagnostics after evaluation."""
        torch = self._torch
        if not self.config.device.startswith("cuda") or not torch.cuda.is_available():
            return {"device": self.config.device}
        return {
            "device": self.config.device,
            "device_name": torch.cuda.get_device_name(self.config.device),
            "memory_allocated_bytes": int(torch.cuda.memory_allocated(self.config.device)),
            "memory_reserved_bytes": int(torch.cuda.memory_reserved(self.config.device)),
            "peak_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated(self.config.device)
            ),
        }

    def encode_texts(self, texts: list[str], *, batch_size: int = 64) -> np.ndarray:
        """Return L2-normalized text embeddings in deterministic input order."""
        if not texts:
            raise ValueError("texts must be non-empty")
        batch_size = _positive_batch_size(batch_size)
        features = []
        torch = self._torch
        for start in range(0, len(texts), batch_size):
            tokens = self.tokenizer(texts[start : start + batch_size]).to(self.config.device)
            with torch.inference_mode(), self._autocast():
                batch_features = self.model.encode_text(tokens)
                batch_features = torch.nn.functional.normalize(batch_features.float(), dim=-1)
            features.append(batch_features.cpu())
        return torch.cat(features, dim=0).numpy().astype(np.float32, copy=False)

    def encode_image_paths(self, paths: list[Path], *, batch_size: int = 32) -> np.ndarray:
        """Load RGB images and return L2-normalized embeddings in input order."""
        if not paths:
            raise ValueError("paths must be non-empty")
        batch_size = _positive_batch_size(batch_size)
        features = []
        torch = self._torch
        for start in range(0, len(paths), batch_size):
            tensors = []
            for path in paths[start : start + batch_size]:
                with Image.open(path) as image:
                    tensors.append(self.preprocess(image.convert("RGB")))
            batch = torch.stack(tensors).to(self.config.device)
            with torch.inference_mode(), self._autocast():
                batch_features = self.model.encode_image(batch)
                batch_features = torch.nn.functional.normalize(batch_features.float(), dim=-1)
            features.append(batch_features.cpu())
        return torch.cat(features, dim=0).numpy().astype(np.float32, copy=False)

    def _autocast(self) -> Any:
        torch = self._torch
        if not self.config.device.startswith("cuda") or self.config.precision == "fp32":
            return nullcontext()
        dtype = torch.bfloat16 if self.config.precision == "amp_bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)


def _positive_batch_size(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("batch_size must be a positive integer")
    return value
