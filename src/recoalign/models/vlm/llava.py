"""Lazy LLaVA-1.5 adapter for pinned local checkpoints."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from datasets.records import SceneRecord
from recoalign.models.vlm.base import BaseVLM, PreparedInput
from synthetic_world.compositional_tasks.tasks import InputSetting


class Llava15VLM(BaseVLM):
    model_id = "llava_1_5_7b"

    def __init__(
        self,
        *,
        model_dir: str | Path,
        backend: Any | None = None,
        auto_load: bool = False,
        tokenizer_path: str | Path | None = None,
        vision_tower_path: str | Path | None = None,
        loader: str = "transformers_llava",
        dtype: str = "float16",
        device_map: str = "auto",
        quantization: str | None = None,
        local_files_only: bool = True,
        generation_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_dir = Path(model_dir)
        self.auto_load = auto_load
        self.dtype = dtype
        self.loader = loader
        self.device_map = device_map
        self.quantization = quantization
        self.local_files_only = local_files_only
        self.generation = dict(generation_config or {})
        self.backend = backend
        if self.backend is None and auto_load:
            backend_class = (
                _LegacyLlavaBackend
                if loader == "legacy_transformers"
                else _TransformersLlavaBackend
            )
            if loader not in {"legacy_transformers", "transformers_llava"}:
                raise ValueError(f"unsupported LLaVA loader: {loader}")
            self.backend = backend_class(
                self.model_dir,
                tokenizer_path=Path(tokenizer_path) if tokenizer_path else self.model_dir,
                vision_tower_path=(str(vision_tower_path) if vision_tower_path else None),
                dtype=dtype,
                device_map=device_map,
                quantization=quantization,
                local_files_only=local_files_only,
                generation_config=self.generation,
            )

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        auto_load: bool = False,
        quantization: str | None = None,
        max_new_tokens: int = 32,
        **kwargs: Any,
    ) -> Llava15VLM:
        path = Path(model_dir)
        if not path.exists():
            raise FileNotFoundError(f"LLaVA checkpoint directory does not exist: {path}")
        generation = dict(kwargs.pop("generation_config", {}))
        generation.setdefault("max_new_tokens", max_new_tokens)
        return cls(
            model_dir=path,
            auto_load=auto_load,
            quantization=quantization,
            generation_config=generation,
            **kwargs,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any], **kwargs: Any) -> Llava15VLM:
        generation = dict(config.get("generation", {}))
        return cls.from_pretrained(
            config["model_path"],
            auto_load=bool(config.get("auto_load", True)),
            tokenizer_path=config.get("tokenizer_path"),
            vision_tower_path=config.get("vision_tower_path"),
            loader=str(config.get("loader", "transformers_llava")),
            dtype=str(config.get("dtype", "float16")),
            device_map=str(config.get("device_map", "auto")),
            quantization=config.get("quantization"),
            local_files_only=bool(config.get("local_files_only", True)),
            max_new_tokens=int(generation.get("max_new_tokens", 32)),
            generation_config=generation,
            **kwargs,
        )

    def load(self) -> Llava15VLM:
        backend = self._require_backend()
        if hasattr(backend, "load"):
            backend.load()
        self._loaded = True
        return self

    def _require_backend(self) -> Any:
        if self.backend is None:
            raise RuntimeError(
                "No LLaVA backend is configured. Set auto_load=true for a pinned local "
                "checkpoint or inject a test backend."
            )
        return self.backend

    @property
    def supports_interface_injection(self) -> bool:
        return self.backend is not None and hasattr(self.backend, "generate_with_interface")

    def generate(self, prompt: str, *, image: str | Path | None = None, **kwargs: Any) -> str:
        self.ensure_loaded()
        generation = {**self.generation, **kwargs}
        return str(self._require_backend().generate(prompt, image=image, **generation))

    def generate_with_interface(
        self,
        prompt: str,
        *,
        image: str | Path,
        interface_checkpoint: str | Path,
        interface_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        backend = self._require_backend()
        if not hasattr(backend, "generate_with_interface"):
            return super().generate_with_interface(
                prompt,
                image=image,
                interface_checkpoint=interface_checkpoint,
                interface_config=interface_config,
                **kwargs,
            )
        self.ensure_loaded()
        return str(
            backend.generate_with_interface(
                prompt,
                image=image,
                interface_checkpoint=interface_checkpoint,
                interface_config=interface_config or {},
                **{**self.generation, **kwargs},
            )
        )

    def encode_image(self, images: Sequence[str | Path]) -> np.ndarray:
        self.ensure_loaded()
        return np.asarray(self._require_backend().encode_image(images))

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        backend = self._require_backend()
        if not hasattr(backend, "encode_text"):
            return super().encode_text(texts)
        self.ensure_loaded()
        return np.asarray(backend.encode_text(texts))

    def encode_visual_tokens(self, images: Sequence[str | Path]) -> np.ndarray:
        self.ensure_loaded()
        backend = self._require_backend()
        if not hasattr(backend, "encode_visual_tokens"):
            raise NotImplementedError(f"{self.model_id} does not expose projected visual tokens")
        return np.asarray(backend.encode_visual_tokens(images))

    def encode_visual_hidden(self, images: Sequence[str | Path]) -> np.ndarray:
        self.ensure_loaded()
        backend = self._require_backend()
        if not hasattr(backend, "encode_visual_hidden"):
            raise NotImplementedError(f"{self.model_id} does not expose visual hidden states")
        return np.asarray(backend.encode_visual_hidden(images))

    def count_tokens(self, text: str) -> int:
        backend = self._require_backend()
        if hasattr(backend, "count_tokens"):
            return int(backend.count_tokens(text))
        return super().count_tokens(text)

    def reason(
        self,
        record: SceneRecord,
        condition: str,
        *,
        prepared_input: PreparedInput | None = None,
        setting: str = InputSetting.NATURAL.value,
        token_match_tolerance: int = 1,
    ) -> str:
        backend = self._require_backend()
        if hasattr(backend, "reason") and prepared_input is None:
            return str(backend.reason(record, condition))
        return super().reason(
            record,
            condition,
            prepared_input=prepared_input,
            setting=setting,
            token_match_tolerance=token_match_tolerance,
        )

    def reason_batch(
        self,
        requests: Sequence[tuple[SceneRecord, str, PreparedInput]],
    ) -> list[str]:
        backend = self._require_backend()
        if (
            self.batch_size <= 1
            or self.inference_cache.enabled
            or not hasattr(backend, "generate_batch")
        ):
            return super().reason_batch(requests)
        self.ensure_loaded()
        predictions: list[str] = []
        for start in range(0, len(requests), self.batch_size):
            chunk = requests[start : start + self.batch_size]
            predictions.extend(
                str(value)
                for value in backend.generate_batch(
                    [prepared.prompt for _record, _condition, prepared in chunk],
                    images=[prepared.image for _record, _condition, prepared in chunk],
                    **self.generation,
                )
            )
        return predictions

    def provenance(self) -> dict[str, Any]:
        return {
            **super().provenance(),
            "model_path": self.model_dir.as_posix(),
            "dtype": self.dtype,
            "loader": self.loader,
            "device_map": self.device_map,
            "quantization": self.quantization,
            "local_files_only": self.local_files_only,
            "generation": self.generation,
            "evidence_role": "scientific_evidence",
        }

    def dry_run(self) -> dict[str, Any]:
        config_path = self.model_dir / "config.json"
        payload = (
            json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        )
        required = ["torch", "transformers", "accelerate", "sentencepiece"]
        normalized_quantization = str(self.quantization or "none").lower()
        cuda_required = normalized_quantization in {"nf4", "int8", "8bit"}
        if cuda_required:
            required.append("bitsandbytes")
        dependencies = {name: importlib.util.find_spec(name) is not None for name in required}
        hardware = {
            "cuda_required": cuda_required,
            "cuda_available": False,
            "cuda_device_count": 0,
            "devices": [],
        }
        if dependencies["torch"]:
            import torch

            hardware["cuda_available"] = bool(torch.cuda.is_available())
            hardware["cuda_device_count"] = int(torch.cuda.device_count())
            hardware["devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
                }
                for index in range(torch.cuda.device_count())
            ]
        hardware_ready = not cuda_required or bool(hardware["cuda_available"])
        checkpoint_files = sorted(
            path.name
            for path in self.model_dir.glob("*")
            if path.is_file() and path.suffix in {".bin", ".safetensors"}
        )
        index_path = self.model_dir / "pytorch_model.bin.index.json"
        checkpoint_format = "unknown"
        if index_path.is_file():
            weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
            keys = tuple(str(key) for key in weight_map)
            checkpoint_format = (
                "legacy_llava"
                if any(key.startswith("model.layers.") for key in keys)
                else "transformers_llava"
                if any(key.startswith("model.language_model.layers.") for key in keys)
                else "unknown"
            )
        loader_compatible = (self.loader, checkpoint_format) in {
            ("legacy_transformers", "legacy_llava"),
            ("transformers_llava", "transformers_llava"),
        }
        processor_ready = False
        processor_error = None
        processor_classes: dict[str, str] = {}
        try:
            backend = self._require_backend()
            if hasattr(backend, "_ensure_processor"):
                processor = backend._ensure_processor()
                processor_classes = {
                    "processor": type(processor).__name__,
                    "tokenizer": type(processor.tokenizer).__name__,
                    "image_processor": type(processor.image_processor).__name__,
                }
            processor_ready = True
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            processor_error = f"{type(exc).__name__}: {exc}"
        return {
            "adapter": self.model_id,
            "checkpoint_exists": self.model_dir.is_dir(),
            "checkpoint_files": checkpoint_files,
            "model_type": payload.get("model_type"),
            "architectures": payload.get("architectures", []),
            "checkpoint_format": checkpoint_format,
            "loader": self.loader,
            "loader_compatible": loader_compatible,
            "dependencies": dependencies,
            "hardware": hardware,
            "hardware_ready": hardware_ready,
            "processor_ready": processor_ready,
            "processor_classes": processor_classes,
            "processor_error": processor_error,
            "runtime_ready": (
                self.model_dir.is_dir()
                and bool(checkpoint_files)
                and all(dependencies.values())
                and hardware_ready
                and processor_ready
                and loader_compatible
            ),
            "weights_loaded": self.loaded,
        }


class _TransformersLlavaBackend:
    def __init__(
        self,
        model_dir: Path,
        *,
        tokenizer_path: Path,
        vision_tower_path: str | Path | None,
        dtype: str,
        device_map: str,
        quantization: str | None,
        local_files_only: bool,
        generation_config: dict[str, Any],
    ) -> None:
        self.model_dir = model_dir
        self.tokenizer_path = tokenizer_path
        self.vision_tower_path = vision_tower_path
        self.dtype = dtype
        self.device_map = device_map
        self.quantization = quantization
        self.local_files_only = local_files_only
        self.generation_config = generation_config
        self.processor: Any | None = None
        self.tokenizer: Any | None = None
        self.model: Any | None = None

    def load(self) -> _TransformersLlavaBackend:
        if self.model is not None:
            return self
        import torch
        from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration

        self._ensure_processor()
        dtype = _torch_dtype(torch, self.dtype)
        kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "local_files_only": self.local_files_only,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        normalized_quantization = str(self.quantization or "none").lower()
        if normalized_quantization == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
        elif normalized_quantization in {"int8", "8bit"}:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif normalized_quantization not in {"none", "fp16", "bf16", "fp32"}:
            raise ValueError(f"unsupported LLaVA quantization: {self.quantization}")
        self.model = LlavaForConditionalGeneration.from_pretrained(self.model_dir, **kwargs)
        self.model.eval()
        return self

    def _ensure_processor(self) -> Any:
        if self.processor is not None:
            return self.processor
        from transformers import AutoTokenizer, CLIPImageProcessor, LlavaProcessor

        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            local_files_only=self.local_files_only,
            use_fast=False,
        )
        try:
            image_processor = CLIPImageProcessor.from_pretrained(
                self.model_dir, local_files_only=self.local_files_only
            )
        except OSError:
            tower = self.vision_tower_path or _vision_tower_from_config(self.model_dir)
            image_processor = CLIPImageProcessor.from_pretrained(
                tower, local_files_only=self.local_files_only
            )
        self.processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)
        self.tokenizer = tokenizer
        return self.processor

    @staticmethod
    def _format_prompt(prompt: str) -> str:
        return f"USER: <image>\n{prompt}\nASSISTANT:"

    def count_tokens(self, prompt: str) -> int:
        self._ensure_processor()
        return len(
            self.tokenizer(self._format_prompt(prompt), add_special_tokens=True)["input_ids"]
        )

    def generate(self, prompt: str, *, image: str | Path | None = None, **kwargs: Any) -> str:
        if image is None:
            raise ValueError("LLaVA structured-reasoning conditions require the paired scene image")
        self.load()
        with Image.open(image) as source:
            pixel_image = source.convert("RGB")
        inputs = self.processor(
            images=pixel_image,
            text=self._format_prompt(prompt),
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generation = _generation_kwargs(self.generation_config, kwargs)
        output = self.model.generate(**inputs, **generation)
        new_tokens = output[:, inputs["input_ids"].shape[1] :]
        return str(self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]).strip()

    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        images: Sequence[str | Path | None],
        **kwargs: Any,
    ) -> list[str]:
        if len(prompts) != len(images) or any(image is None for image in images):
            raise ValueError("LLaVA batched prompts require one paired image per prompt")
        self.load()
        loaded = []
        for path in images:
            with Image.open(path) as source:
                loaded.append(source.convert("RGB"))
        inputs = self.processor(
            images=loaded,
            text=[self._format_prompt(prompt) for prompt in prompts],
            return_tensors="pt",
            padding=True,
        )
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generation = _generation_kwargs(self.generation_config, kwargs)
        output = self.model.generate(**inputs, **generation)
        new_tokens = output[:, inputs["input_ids"].shape[1] :]
        return [
            str(value).strip()
            for value in self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        ]

    def encode_image(self, images: Sequence[str | Path]) -> np.ndarray:
        self.load()
        loaded = []
        for path in images:
            with Image.open(path) as source:
                loaded.append(source.convert("RGB"))
        pixel_values = self.processor(images=loaded, return_tensors="pt")["pixel_values"]
        device = next(self.model.parameters()).device
        with __import__("torch").inference_mode():
            features = self.model.get_image_features(pixel_values=pixel_values.to(device))
        return features.detach().float().cpu().numpy()

    def encode_visual_tokens(self, images: Sequence[str | Path]) -> np.ndarray:
        """Use the frozen CLIP/projector path as the projected-token diagnostic."""

        return self.encode_image(images)


class _LegacyLlavaBackend(_TransformersLlavaBackend):
    """Execute original LLaVA-1.5 checkpoints without silently remapping wrong keys."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.vision_model: Any | None = None
        self.projector: Any | None = None
        self.vision_feature_layer = -2

    def load(self) -> _LegacyLlavaBackend:
        if self.model is not None:
            return self
        import torch
        from transformers import (
            BitsAndBytesConfig,
            CLIPVisionModel,
            LlamaConfig,
            LlamaForCausalLM,
        )

        self._ensure_processor()
        dtype = _torch_dtype(torch, self.dtype)
        language_config = LlamaConfig.from_pretrained(
            self.model_dir, local_files_only=self.local_files_only
        )
        raw_config = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        self.vision_feature_layer = int(raw_config.get("mm_vision_select_layer", -2))
        kwargs: dict[str, Any] = {
            "config": language_config,
            "device_map": self.device_map,
            "local_files_only": self.local_files_only,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        normalized_quantization = str(self.quantization or "none").lower()
        if normalized_quantization == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
        elif normalized_quantization in {"int8", "8bit"}:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif normalized_quantization not in {"none", "fp16", "bf16", "fp32"}:
            raise ValueError(f"unsupported LLaVA quantization: {self.quantization}")
        self.model = LlamaForCausalLM.from_pretrained(self.model_dir, **kwargs)
        self.model.eval()
        device = next(self.model.parameters()).device
        tower = self.vision_tower_path or _vision_tower_from_config(self.model_dir)
        self.vision_model = CLIPVisionModel.from_pretrained(
            tower,
            local_files_only=self.local_files_only,
            torch_dtype=dtype,
        ).to(device)
        self.vision_model.eval()
        hidden_size = int(language_config.hidden_size)
        vision_size = int(self.vision_model.config.hidden_size)
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(vision_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, hidden_size),
        ).to(device=device, dtype=dtype)
        projector_state = torch.load(
            self.model_dir / "mm_projector.bin",
            map_location="cpu",
            weights_only=True,
        )
        mapped = {
            key.removeprefix("model.mm_projector."): value for key, value in projector_state.items()
        }
        self.projector.load_state_dict(mapped, strict=True)
        self.projector.eval()
        return self

    def count_tokens(self, prompt: str) -> int:
        self._ensure_processor()
        formatted = self._format_prompt(prompt)
        before, after = formatted.split("<image>", maxsplit=1)
        prefix = self.tokenizer(before, add_special_tokens=True)["input_ids"]
        suffix = self.tokenizer(after, add_special_tokens=False)["input_ids"]
        image_tokens = int(
            getattr(self.processor.image_processor, "crop_size", {}).get("height", 336)
        )
        patch_size = 14
        return len(prefix) + (image_tokens // patch_size) ** 2 + len(suffix)

    def generate(self, prompt: str, *, image: str | Path | None = None, **kwargs: Any) -> str:
        if image is None:
            raise ValueError("LLaVA structured-reasoning conditions require the paired scene image")
        import torch

        self.load()
        embeddings = self._multimodal_embeddings(prompt, image)
        attention = torch.ones(embeddings.shape[:2], dtype=torch.long, device=embeddings.device)
        generation = _generation_kwargs(self.generation_config, kwargs)
        generation.update(
            {
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
        )
        with torch.inference_mode():
            output = self.model.generate(
                inputs_embeds=embeddings,
                attention_mask=attention,
                **generation,
            )
        return str(self.tokenizer.decode(output[0], skip_special_tokens=True)).strip()

    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        images: Sequence[str | Path | None],
        **kwargs: Any,
    ) -> list[str]:
        return [
            self.generate(prompt, image=image, **kwargs)
            for prompt, image in zip(prompts, images, strict=True)
        ]

    def encode_image(self, images: Sequence[str | Path]) -> np.ndarray:
        import torch

        self.load()
        features = []
        for image in images:
            with Image.open(image) as source:
                pixel = source.convert("RGB")
            values = self.processor.image_processor(images=pixel, return_tensors="pt")[
                "pixel_values"
            ]
            device = next(self.model.parameters()).device
            dtype = next(self.projector.parameters()).dtype
            with torch.inference_mode():
                vision = self.vision_model(
                    pixel_values=values.to(device=device, dtype=dtype),
                    output_hidden_states=True,
                ).hidden_states[self.vision_feature_layer][:, 1:]
                projected = self.projector(vision).mean(dim=1)
            features.append(projected[0].detach().float().cpu().numpy())
        return np.stack(features)

    def _multimodal_embeddings(self, prompt: str, image: str | Path) -> Any:
        import torch

        formatted = self._format_prompt(prompt)
        before, after = formatted.split("<image>", maxsplit=1)
        device = next(self.model.parameters()).device
        prefix_ids = self.tokenizer(before, add_special_tokens=True, return_tensors="pt")[
            "input_ids"
        ].to(device)
        suffix_ids = self.tokenizer(after, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].to(device)
        token_embeddings = self.model.get_input_embeddings()
        prefix = token_embeddings(prefix_ids)
        suffix = token_embeddings(suffix_ids)
        with Image.open(image) as source:
            pixel = source.convert("RGB")
        values = self.processor.image_processor(images=pixel, return_tensors="pt")["pixel_values"]
        dtype = next(self.projector.parameters()).dtype
        with torch.inference_mode():
            vision = self.vision_model(
                pixel_values=values.to(device=device, dtype=dtype),
                output_hidden_states=True,
            ).hidden_states[self.vision_feature_layer][:, 1:]
            image_features = self.projector(vision)
        return torch.cat((prefix.to(dtype), image_features.to(dtype), suffix.to(dtype)), dim=1)


def _generation_kwargs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    values = {**base, **override}
    do_sample = bool(values.get("do_sample", False))
    result: dict[str, Any] = {
        "do_sample": do_sample,
        "max_new_tokens": int(values.get("max_new_tokens", 32)),
    }
    if do_sample:
        result["temperature"] = float(values.get("temperature", 1.0))
    return result


def _torch_dtype(torch: Any, value: str) -> Any:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if value.lower() not in mapping:
        raise ValueError(f"unsupported dtype: {value}")
    return mapping[value.lower()]


def _vision_tower_from_config(model_dir: Path) -> str:
    payload = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tower = payload.get("mm_vision_tower")
    if not isinstance(tower, str) or not tower:
        raise ValueError("LLaVA config does not declare mm_vision_tower")
    return tower


__all__ = ["Llava15VLM"]
