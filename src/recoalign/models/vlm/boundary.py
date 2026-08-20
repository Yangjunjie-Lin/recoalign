"""Common injectable boundary for adapter-ready VLM families."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from recoalign.models.vlm.base import BaseVLM


class AdapterBoundaryVLM(BaseVLM):
    model_id = "adapter-boundary"

    def __init__(
        self,
        *,
        model_path: str | Path | None,
        backend: Any | None = None,
        auto_load: bool = False,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        generation_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_path = Path(model_path) if model_path else None
        self.backend = backend
        self.auto_load = auto_load
        self.dtype = dtype
        self.device_map = device_map
        self.generation = dict(generation_config or {})

    @classmethod
    def from_config(cls, config: dict[str, Any], **kwargs: Any) -> AdapterBoundaryVLM:
        return cls(
            model_path=config.get("model_path"),
            auto_load=bool(config.get("auto_load", False)),
            dtype=str(config.get("dtype", "bfloat16")),
            device_map=str(config.get("device_map", "auto")),
            generation_config=dict(config.get("generation", {})),
            **kwargs,
        )

    def load(self) -> AdapterBoundaryVLM:
        if self.backend is None:
            self.backend = self._load_backend()
        if hasattr(self.backend, "load"):
            self.backend.load()
        self._loaded = True
        return self

    def _load_backend(self) -> Any:
        raise RuntimeError(
            f"{self.model_id} adapter boundary is ready, but no pinned family runtime/checkpoint "
            "is configured. Inject a backend or provide the registered local model package."
        )

    @property
    def supports_interface_injection(self) -> bool:
        return self.backend is not None and hasattr(self.backend, "generate_with_interface")

    def generate(self, prompt: str, *, image: str | Path | None = None, **kwargs: Any) -> str:
        self.ensure_loaded()
        return str(self.backend.generate(prompt, image=image, **{**self.generation, **kwargs}))

    def generate_with_interface(
        self,
        prompt: str,
        *,
        image: str | Path,
        interface_checkpoint: str | Path,
        interface_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        self.ensure_loaded()
        if not hasattr(self.backend, "generate_with_interface"):
            return super().generate_with_interface(
                prompt,
                image=image,
                interface_checkpoint=interface_checkpoint,
                interface_config=interface_config,
                **kwargs,
            )
        return str(
            self.backend.generate_with_interface(
                prompt,
                image=image,
                interface_checkpoint=interface_checkpoint,
                interface_config=interface_config or {},
                **{**self.generation, **kwargs},
            )
        )

    def encode_image(self, images: Sequence[str | Path]) -> np.ndarray:
        self.ensure_loaded()
        if not hasattr(self.backend, "encode_image"):
            raise NotImplementedError(f"{self.model_id} runtime does not expose image embeddings")
        return np.asarray(self.backend.encode_image(images))

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        self.ensure_loaded()
        if not hasattr(self.backend, "encode_text"):
            return super().encode_text(texts)
        return np.asarray(self.backend.encode_text(texts))

    def count_tokens(self, text: str) -> int:
        if self.backend is not None and hasattr(self.backend, "count_tokens"):
            return int(self.backend.count_tokens(text))
        return super().count_tokens(text)

    def dry_run(self) -> dict[str, Any]:
        dependencies = {
            name: importlib.util.find_spec(name) is not None
            for name in ("torch", "transformers", "accelerate")
        }
        checkpoint_exists = self.model_path is not None and self.model_path.exists()
        return {
            "adapter": self.model_id,
            "checkpoint_configured": self.model_path is not None,
            "checkpoint_exists": checkpoint_exists,
            "dependencies": dependencies,
            "adapter_ready": True,
            "runtime_ready": False,
            "weights_loaded": self.loaded,
        }

    def provenance(self) -> dict[str, Any]:
        return {
            **super().provenance(),
            "model_path": self.model_path.as_posix() if self.model_path else None,
            "dtype": self.dtype,
            "device_map": self.device_map,
            "generation": self.generation,
        }


__all__ = ["AdapterBoundaryVLM"]
