from __future__ import annotations

import hashlib
import json
import platform
import shutil
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm

from .config import ExperimentConfig
from .data import read_jsonl


class FrozenRepresentationExtractor(ABC):
    """Frozen hook-based extractor for the two representation boundaries."""

    processor: Any
    device: torch.device
    identity: dict[str, Any]
    zv: torch.Tensor | None
    za: torch.Tensor | None

    @abstractmethod
    def extract(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class ClipProjectionExtractor(FrozenRepresentationExtractor):
    def __init__(self, model_name: str, device: torch.device) -> None:
        try:
            from huggingface_hub import hf_hub_download
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers is required; run pip install -r requirements.txt"
            ) from exc
        config_path = Path(hf_hub_download(repo_id=model_name, filename="config.json"))
        model_commit = config_path.parent.name
        self.processor = CLIPImageProcessor.from_pretrained(model_name, revision=model_commit)
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            model_name, revision=model_commit
        )
        self.model.eval().requires_grad_(False).to(device)
        self.device = device
        self.zv = None
        self.za = None
        self._handles = [
            self.model.vision_model.register_forward_hook(self._vision_hook),
            self.model.visual_projection.register_forward_hook(self._projection_hook),
        ]
        self.identity = {
            "backend": "clip_trained_projection",
            "model_name": model_name,
            "model_commit": model_commit,
            "zv_dim": int(self.model.config.hidden_size),
            "za_dim": int(self.model.config.projection_dim),
            "pooling": "CLIP pooled CLS output",
            "zl_available": False,
            "note": (
                "Zv is CLIP vision_model pooled output; Za is its trained visual_projection output."
            ),
        }

    def _vision_hook(self, _module: nn.Module, _inputs: Any, output: Any) -> None:
        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            pooled = output[1]
        self.zv = pooled.detach()

    def _projection_hook(self, _module: nn.Module, _inputs: Any, output: Any) -> None:
        self.za = output.detach()

    def extract(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.zv = None
        self.za = None
        self.model(pixel_values=pixel_values, return_dict=True)
        if self.zv is None or self.za is None:
            raise RuntimeError("CLIP representation hooks did not fire")
        return self.zv, self.za

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


class LlavaProjectorExtractor(FrozenRepresentationExtractor):
    """The real LLaVA-1.5 vision tower and official MLP projector, without the 7B LLM."""

    def __init__(self, repo: str, vision_model_name: str, device: torch.device) -> None:
        try:
            from huggingface_hub import hf_hub_download
            from transformers import CLIPImageProcessor, CLIPVisionModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("transformers and huggingface_hub are required") from exc

        config_path = Path(hf_hub_download(repo_id=repo, filename="config.json"))
        llava_config = json.loads(config_path.read_text(encoding="utf-8"))
        configured_tower = str(llava_config["mm_vision_tower"])
        if configured_tower != vision_model_name:
            raise RuntimeError(
                f"Requested vision tower {vision_model_name!r} does not match "
                f"LLaVA config {configured_tower!r}"
            )
        if llava_config.get("mm_projector_type") != "mlp2x_gelu":
            raise RuntimeError(
                "This loader supports the official LLaVA-1.5 mlp2x_gelu projector only"
            )
        if llava_config.get("mm_vision_select_feature") != "patch":
            raise RuntimeError("Expected LLaVA-1.5 to select patch tokens")

        llava_commit = config_path.parent.name
        vision_config_path = Path(
            hf_hub_download(repo_id=vision_model_name, filename="config.json")
        )
        vision_commit = vision_config_path.parent.name
        self.processor = CLIPImageProcessor.from_pretrained(
            vision_model_name, revision=vision_commit
        )
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.vision_model = CLIPVisionModel.from_pretrained(
            vision_model_name, revision=vision_commit, torch_dtype=dtype
        )
        self.vision_model.eval().requires_grad_(False).to(device)

        input_dim = int(llava_config["mm_hidden_size"])
        output_dim = int(llava_config["hidden_size"])
        self.projector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        projector_path = hf_hub_download(
            repo_id=repo, filename="mm_projector.bin", revision=llava_commit
        )
        state = torch.load(projector_path, map_location="cpu", weights_only=True)
        prefix = "model.mm_projector."
        projector_state = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        missing, unexpected = self.projector.load_state_dict(projector_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Invalid LLaVA projector state: missing={missing}, unexpected={unexpected}"
            )
        self.projector.eval().requires_grad_(False).to(device=device, dtype=dtype)

        self.device = device
        self.select_layer = int(llava_config["mm_vision_select_layer"])
        self.zv = None
        self.za = None
        self._selected_tokens: torch.Tensor | None = None
        vision_core = getattr(self.vision_model, "vision_model", self.vision_model)
        # These hooks capture the exact projector input/output. Token means give
        # fixed-size vectors without learning another aggregation layer.
        self._handles = [
            vision_core.encoder.layers[self.select_layer].register_forward_hook(
                self._vision_layer_hook
            ),
            self.projector.register_forward_pre_hook(self._projector_input_hook),
            self.projector.register_forward_hook(self._projector_output_hook),
        ]
        self.identity = {
            "backend": "llava_1_5_7b_vision_projector",
            "model_name": repo,
            "vision_model_name": vision_model_name,
            "model_commit": llava_commit,
            "vision_model_commit": vision_commit,
            "zv_dim": input_dim,
            "za_dim": output_dim,
            "vision_select_layer": self.select_layer,
            "vision_select_feature": "patch",
            "pooling": "mean over selected spatial patch tokens after each boundary",
            "zl_available": False,
            "note": (
                "Actual LLaVA-1.5-7B vision tower and official mm_projector.bin. The 7B LLM is not "
                "loaded, so Zl is unavailable; this isolates Zv-to-Za within 6 GB VRAM."
            ),
        }

    def _vision_layer_hook(self, _module: nn.Module, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        self._selected_tokens = hidden[:, 1:, :]

    def _projector_input_hook(self, _module: nn.Module, inputs: Any) -> None:
        self.zv = inputs[0].detach().mean(dim=1)

    def _projector_output_hook(
        self, _module: nn.Module, _inputs: Any, output: torch.Tensor
    ) -> None:
        self.za = output.detach().mean(dim=1)

    def extract(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.zv = None
        self.za = None
        self._selected_tokens = None
        self.vision_model(pixel_values=pixel_values, return_dict=True)
        if self._selected_tokens is None:
            raise RuntimeError("LLaVA vision-layer hook did not fire")
        self.projector(self._selected_tokens)
        if self.zv is None or self.za is None:
            raise RuntimeError("LLaVA representation hooks did not fire")
        return self.zv, self.za

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if not torch.cuda.is_available() and shutil.which("nvidia-smi") is not None:
            warnings.warn(
                "An NVIDIA driver is present, but this Python cannot access CUDA. "
                "Auto will use the CLIP CPU fallback. Install a CUDA-enabled PyTorch "
                "wheel to select the preferred LLaVA vision-projector path.",
                stacklevel=2,
            )
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but this Python has a CPU-only PyTorch build. "
            "Install a CUDA-enabled PyTorch wheel, then verify torch.cuda.is_available()."
        )
    return device


def _load_backend(config: ExperimentConfig, device: torch.device) -> FrozenRepresentationExtractor:
    backend = config.backend
    if backend == "auto":
        backend = "llava" if device.type == "cuda" else "clip"
    if backend == "llava":
        return LlavaProjectorExtractor(config.llava_repo, config.llava_vision_model, device)
    return ClipProjectionExtractor(config.model_name, device)


def _metadata_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_manifest(
    extractor: FrozenRepresentationExtractor, metadata_path: Path, count: int
) -> dict[str, Any]:
    return {
        "feature_format_version": 2,
        "backend": extractor.identity["backend"],
        "model_name": extractor.identity["model_name"],
        "model_commit": extractor.identity["model_commit"],
        "vision_model_commit": extractor.identity.get("vision_model_commit"),
        "metadata_sha256": _metadata_digest(metadata_path),
        "count": count,
    }


def _cache_is_valid(output_dir: Path, expected: dict[str, Any]) -> bool:
    manifest_path = output_dir / "manifest.json"
    paths = (
        manifest_path,
        output_dir / "zv" / "features.npy",
        output_dir / "za" / "features.npy",
        output_dir / "index.jsonl",
    )
    if not all(path.exists() for path in paths):
        return False
    try:
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(actual.get(key) == value for key, value in expected.items())


def _write_index(output_dir: Path, records: list[dict[str, Any]]) -> None:
    path = output_dir / "index.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row_index, record in enumerate(records):
            index_record = {
                "image_id": record["image_id"],
                "row_index": row_index,
                "Zv": {"file": "zv/features.npy", "row": row_index},
                "Za": {"file": "za/features.npy", "row": row_index},
                "Zl": None,
                "semantic_label": record["semantic_label"],
            }
            handle.write(json.dumps(index_record, sort_keys=True) + "\n")


def _extract_records(
    *,
    config: ExperimentConfig,
    records: list[dict[str, Any]],
    metadata_path: Path,
    output_dir: Path,
    extractor: FrozenRepresentationExtractor,
) -> None:
    expected = _expected_manifest(extractor, metadata_path, len(records))
    if not config.force_features and _cache_is_valid(output_dir, expected):
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "zv").mkdir(parents=True, exist_ok=True)
    (output_dir / "za").mkdir(parents=True, exist_ok=True)

    zv_batches: list[np.ndarray] = []
    za_batches: list[np.ndarray] = []
    amp_enabled = extractor.device.type == "cuda"
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(records), config.batch_size), desc=f"Extract {output_dir.name}"
        ):
            batch = records[start : start + config.batch_size]
            images: list[Image.Image] = []
            try:
                for record in batch:
                    images.append(Image.open(config.dataset_dir / record["image"]).convert("RGB"))
                inputs = extractor.processor(images=images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(extractor.device, non_blocking=amp_enabled)
                with torch.autocast(
                    device_type=extractor.device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    zv_batch, za_batch = extractor.extract(pixel_values)
                zv_batches.append(zv_batch.float().cpu().numpy())
                za_batches.append(za_batch.float().cpu().numpy())
            finally:
                for image in images:
                    image.close()

    zv = np.concatenate(zv_batches, axis=0).astype(np.float32, copy=False)
    za = np.concatenate(za_batches, axis=0).astype(np.float32, copy=False)
    if zv.shape[0] != len(records) or za.shape[0] != len(records):
        raise RuntimeError("Feature row count does not match metadata")
    np.save(output_dir / "zv" / "features.npy", zv, allow_pickle=False)
    np.save(output_dir / "za" / "features.npy", za, allow_pickle=False)
    _write_index(output_dir, records)

    manifest = {
        **expected,
        **extractor.identity,
        "zv_shape": list(zv.shape),
        "za_shape": list(za.shape),
        "zv_dtype": str(zv.dtype),
        "za_dtype": str(za.dtype),
        "device": str(extractor.device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def extract_all_features(config: ExperimentConfig) -> dict[str, Any]:
    metadata_path = config.dataset_dir / "metadata.jsonl"
    control_metadata_path = config.dataset_dir / "control_metadata.jsonl"
    records = read_jsonl(metadata_path)
    control_records = read_jsonl(control_metadata_path)
    device = _resolve_device(config.device)
    extractor = _load_backend(config, device)
    try:
        _extract_records(
            config=config,
            records=records,
            metadata_path=metadata_path,
            output_dir=config.features_dir / "main",
            extractor=extractor,
        )
        _extract_records(
            config=config,
            records=control_records,
            metadata_path=control_metadata_path,
            output_dir=config.features_dir / "control",
            extractor=extractor,
        )
        return {**extractor.identity, "device": str(device)}
    finally:
        extractor.close()


def load_features(config: ExperimentConfig, subset: str, stage: str) -> np.ndarray:
    stage_name = stage.lower()
    if stage_name not in {"zv", "za"}:
        raise ValueError(f"Unknown stage: {stage}")
    return np.load(config.features_dir / subset / stage_name / "features.npy", mmap_mode="r")
