"""Ground-truth-blind primary conditional-likelihood measurement."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from recoalign.models.vlm.base import PreparedInput

from .inventory import PRIMARY_METHOD, REGISTERED_CHOICE_IDS

FROZEN_TRANSFORMERS_VERSION = "4.52.4"
FROZEN_TOKENIZER = Path(__file__).resolve().parents[3] / "outputs/models/llava-v1.5-7b"
_TOKENIZER_PATCH_MARKER = "_recoalign_pivot_a3p_no_prefix_space"


class ChoiceLikelihoodBackend(Protocol):
    def score_choice_log_likelihoods(
        self, prepared_input: PreparedInput, choice_ids: Sequence[str]
    ) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class PrimaryScore:
    selected_option_id: str
    choice_log_likelihoods: dict[str, float]
    tied_maximum: bool
    tie_state: str
    valid_measurement: bool = True
    method: str = PRIMARY_METHOD

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_option_id": self.selected_option_id,
            "choice_log_likelihoods": dict(self.choice_log_likelihoods),
            "choice_ids": list(REGISTERED_CHOICE_IDS),
            "tied_maximum": self.tied_maximum,
            "tie_state": self.tie_state,
            "valid_measurement": self.valid_measurement,
            "scoring_method": self.method,
        }


def score_registered_options(
    prepared_input: PreparedInput,
    choice_ids: Sequence[str],
    *,
    backend: ChoiceLikelihoodBackend,
) -> PrimaryScore:
    """Score exactly four registered IDs without accepting any truth or answer argument."""

    registered = tuple(str(value) for value in choice_ids)
    if registered != REGISTERED_CHOICE_IDS:
        raise ValueError(f"choice IDs must be exactly {REGISTERED_CHOICE_IDS}")
    raw = backend.score_choice_log_likelihoods(prepared_input, registered)
    if set(raw) != set(registered) or len(raw) != 4:
        raise ValueError("likelihood backend must return exactly four registered scores")
    scores = {choice: float(raw[choice]) for choice in registered}
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("all four choice log likelihoods must be finite")
    maximum = max(scores.values())
    winners = [choice for choice in registered if scores[choice] == maximum]
    tied = len(winners) > 1
    return PrimaryScore(
        selected_option_id=winners[0],
        choice_log_likelihoods=scores,
        tied_maximum=tied,
        tie_state="numeric_order_tiebreak" if tied else "unique_maximum",
    )


def validate_tokenizer_contract(tokenizer: Any) -> dict[str, Any]:
    prefix = "FINAL_CHOICE="
    prefix_ids = _token_ids(tokenizer, prefix)
    options = []
    for choice in REGISTERED_CHOICE_IDS:
        standalone = _token_ids(tokenizer, choice)
        combined = _token_ids(tokenizer, prefix + choice)
        suffix = combined[len(prefix_ids) :] if combined[: len(prefix_ids)] == prefix_ids else []
        options.append(
            {
                "choice_id": choice,
                "standalone_token_ids": standalone,
                "prefix_context_token_ids": suffix,
                "single_token": len(standalone) == 1,
                "prefix_boundary_stable": suffix == standalone,
                "passed": len(standalone) == 1 and suffix == standalone,
            }
        )
    ids = [row["standalone_token_ids"][0] for row in options if row["standalone_token_ids"]]
    return {
        "completion_prefix": prefix,
        "choice_ids": list(REGISTERED_CHOICE_IDS),
        "options": options,
        "unique_token_ids": len(ids) == len(set(ids)) == 4,
        "add_prefix_space": getattr(tokenizer, "add_prefix_space", None),
        "passed": all(row["passed"] for row in options)
        and len(ids) == len(set(ids)) == 4
        and getattr(tokenizer, "add_prefix_space", None) is False,
    }


def configure_frozen_tokenizer_runtime() -> dict[str, Any]:
    """Apply the parent-frozen no-leading-space tokenizer runtime compatibility."""

    import transformers
    from transformers import AutoTokenizer

    if transformers.__version__ != FROZEN_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"PIVOT_EXP_A3P requires Transformers {FROZEN_TRANSFORMERS_VERSION}, "
            f"observed {transformers.__version__}"
        )
    descriptor = AutoTokenizer.__dict__["from_pretrained"]
    function = descriptor.__func__
    if getattr(function, _TOKENIZER_PATCH_MARKER, False):
        return {
            "configured": True,
            "already_configured": True,
            "transformers_version": transformers.__version__,
            "tokenizer_path": str(FROZEN_TOKENIZER),
            "add_prefix_space": False,
        }
    original = function

    def frozen_from_pretrained(
        cls: type[Any],
        pretrained_model_name_or_path: str | Path,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        candidate = Path(str(pretrained_model_name_or_path))
        try:
            is_frozen = candidate.resolve() == FROZEN_TOKENIZER.resolve()
        except OSError:
            is_frozen = False
        if is_frozen:
            kwargs["add_prefix_space"] = False
        tokenizer = original(cls, pretrained_model_name_or_path, *inputs, **kwargs)
        if is_frozen:
            if not hasattr(tokenizer, "add_prefix_space"):
                raise TypeError("frozen tokenizer must expose add_prefix_space")
            tokenizer.add_prefix_space = False
        return tokenizer

    setattr(frozen_from_pretrained, _TOKENIZER_PATCH_MARKER, True)
    AutoTokenizer.from_pretrained = classmethod(frozen_from_pretrained)
    return {
        "configured": True,
        "already_configured": False,
        "transformers_version": transformers.__version__,
        "tokenizer_path": str(FROZEN_TOKENIZER),
        "add_prefix_space": False,
    }


class ImageFeatureCache:
    """Disk-backed exact projected-image features for staged all-GPU execution."""

    def __init__(self, root: str | Path, *, model_sha256: str, memory_entries: int = 12) -> None:
        self.root = Path(root) / model_sha256
        self.memory_entries = int(memory_entries)
        self._memory: OrderedDict[str, Any] = OrderedDict()

    def get(self, image_sha256: str) -> Any | None:
        if image_sha256 in self._memory:
            tensor = self._memory.pop(image_sha256)
            self._memory[image_sha256] = tensor
            return tensor
        path = self._path(image_sha256)
        if not path.is_file():
            return None
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=True)
        tensor = payload["image_features"]
        if (
            payload.get("image_sha256") != image_sha256
            or tensor.ndim != 3
            or tensor.shape[0] != 1
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError("staged image-feature cache integrity failure")
        self._remember(image_sha256, tensor)
        return tensor

    def put(self, image_sha256: str, tensor: Any) -> None:
        import torch

        cpu = tensor.detach().to(device="cpu").contiguous()
        if cpu.ndim != 3 or cpu.shape[0] != 1 or not bool(torch.isfinite(cpu).all()):
            raise ValueError("only finite projected image-token tensors may be cached")
        existing = self.get(image_sha256)
        if existing is not None:
            if not torch.equal(existing, cpu):
                raise ValueError("staged image-feature cache drift")
            return
        path = self._path(image_sha256)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".tmp", dir=path.parent
        )
        os.close(descriptor)
        try:
            torch.save(
                {"image_sha256": image_sha256, "image_features": cpu},
                temporary_name,
            )
            with open(temporary_name, "r+b") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()
        self._remember(image_sha256, cpu)

    def _path(self, image_sha256: str) -> Path:
        return self.root / image_sha256[:2] / f"{image_sha256}.pt"

    def _remember(self, image_sha256: str, tensor: Any) -> None:
        self._memory.pop(image_sha256, None)
        self._memory[image_sha256] = tensor
        while len(self._memory) > self.memory_entries:
            self._memory.popitem(last=False)


def prepare_staged_gpu_image_features(
    model: Any,
    *,
    images_by_sha256: Mapping[str, str],
    feature_cache: ImageFeatureCache,
    progress_label: str,
) -> dict[str, Any]:
    """Compute vision features on CUDA, then reload language weights and release vision."""

    import gc
    import time

    import torch

    model.ensure_loaded()
    backend = model._require_backend()  # noqa: SLF001
    if not all(
        value is not None for value in (backend.model, backend.vision_model, backend.projector)
    ):
        raise ValueError("staged feature preparation requires the complete frozen backend")
    if any("cpu" in str(parameter.device) for parameter in backend.vision_model.parameters()):
        raise ValueError("vision feature preparation refuses CPU parameter execution")
    if any("cpu" in str(parameter.device) for parameter in backend.projector.parameters()):
        raise ValueError("projector feature preparation refuses CPU parameter execution")
    started = time.perf_counter()
    backend.model = None
    gc.collect()
    torch.cuda.empty_cache()
    device = next(backend.vision_model.parameters()).device
    dtype = next(backend.projector.parameters()).dtype
    computed = 0
    reused = 0
    feature_shapes = set()
    ordered = sorted(images_by_sha256.items())
    for index, (image_sha, image_path) in enumerate(ordered, start=1):
        cached = feature_cache.get(image_sha)
        if cached is None:
            with Image.open(image_path) as source:
                pixel = source.convert("RGB")
            values = backend.processor.image_processor(images=pixel, return_tensors="pt")[
                "pixel_values"
            ]
            with torch.inference_mode():
                vision = backend.vision_model(
                    pixel_values=values.to(device=device, dtype=dtype),
                    output_hidden_states=True,
                ).hidden_states[backend.vision_feature_layer][:, 1:]
                features = backend.projector(vision)
            feature_cache.put(image_sha, features)
            cached = features.detach().cpu()
            computed += 1
            del vision, features, values
        else:
            reused += 1
        feature_shapes.add(tuple(int(value) for value in cached.shape))
        if index % 10 == 0 or index == len(ordered):
            print(
                f"{progress_label} image_features={index}/{len(ordered)} "
                f"computed={computed} reused={reused} "
                f"elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )
    backend.vision_model = None
    backend.projector = None
    gc.collect()
    torch.cuda.empty_cache()
    backend.load()
    language_devices = sorted({str(parameter.device) for parameter in backend.model.parameters()})
    reloaded_vision_devices = sorted(
        {str(parameter.device) for parameter in backend.vision_model.parameters()}
    )
    reloaded_projector_devices = sorted(
        {str(parameter.device) for parameter in backend.projector.parameters()}
    )
    backend.vision_model = None
    backend.projector = None
    gc.collect()
    torch.cuda.empty_cache()
    if not language_devices or any("cpu" in name for name in language_devices):
        raise ValueError("language scoring refuses CPU parameter execution")
    return {
        "strategy": "staged_gpu_vision_feature_cache_then_nf4_language_forward_v1",
        "image_count": len(ordered),
        "computed_features": computed,
        "reused_features": reused,
        "feature_shapes": sorted(feature_shapes),
        "vision_execution_devices": [str(device)],
        "projector_execution_devices": [str(device)],
        "language_execution_devices": language_devices,
        "reloaded_vision_devices_before_release": reloaded_vision_devices,
        "reloaded_projector_devices_before_release": reloaded_projector_devices,
        "vision_modules_released_before_language_scoring": True,
        "cpu_model_parameter_execution": False,
        "no_cpu_fallback": True,
        "elapsed_seconds": time.perf_counter() - started,
        "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
        "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved()),
        "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


class LlavaPrimaryLikelihoodBackend:
    """Frozen LLaVA next-token likelihood adapter; no generation surface is exposed."""

    def __init__(
        self,
        model: Any,
        *,
        feature_cache: ImageFeatureCache,
        image_sha256_by_path: Mapping[str, str],
    ) -> None:
        self.model = model
        self.feature_cache = feature_cache
        self.image_sha256_by_path = {
            str(Path(path).resolve()): digest for path, digest in image_sha256_by_path.items()
        }

    def score_choice_log_likelihoods(
        self, prepared_input: PreparedInput, choice_ids: Sequence[str]
    ) -> Mapping[str, float]:
        self.model.ensure_loaded()
        backend = self.model._require_backend()  # noqa: SLF001
        tokenizer_report = validate_tokenizer_contract(backend.tokenizer)
        if not tokenizer_report["passed"]:
            raise ValueError("frozen tokenizer contract failed at scoring boundary")
        token_ids = {
            row["choice_id"]: int(row["standalone_token_ids"][0])
            for row in tokenizer_report["options"]
        }
        logits = self._next_token_logits(backend, prepared_input)
        import torch

        log_probabilities = torch.log_softmax(logits.float(), dim=-1)
        return {
            choice: float(log_probabilities[token_id].detach().cpu())
            for choice, token_id in token_ids.items()
            if choice in choice_ids
        }

    def _next_token_logits(self, backend: Any, prepared_input: PreparedInput) -> Any:
        import torch

        if prepared_input.image is None:
            raise ValueError("primary scoring requires a frozen registered image")
        backend.model.eval()
        if hasattr(backend, "vision_model") and backend.vision_model is not None:
            backend.vision_model.eval()
        if hasattr(backend, "projector") and backend.projector is not None:
            backend.projector.eval()
        if hasattr(backend, "_multimodal_embeddings"):
            normalized = str(Path(prepared_input.image).resolve())
            image_sha = self.image_sha256_by_path.get(normalized)
            if image_sha is None:
                raise ValueError("image has no frozen feature-cache identity")
            image_features = self.feature_cache.get(image_sha)
            if image_features is None:
                raise ValueError("registered image feature is missing from staged GPU cache")
            formatted = backend._format_prompt(prepared_input.prompt)  # noqa: SLF001
            before, after = formatted.split("<image>", maxsplit=1)
            device = next(backend.model.parameters()).device
            prefix_ids = backend.tokenizer(before, add_special_tokens=True, return_tensors="pt")[
                "input_ids"
            ].to(device)
            suffix_ids = backend.tokenizer(after, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ].to(device)
            token_embeddings = backend.model.get_input_embeddings()
            dtype = token_embeddings.weight.dtype
            embeddings = torch.cat(
                (
                    token_embeddings(prefix_ids).to(dtype),
                    image_features.to(device=device, dtype=dtype),
                    token_embeddings(suffix_ids).to(dtype),
                ),
                dim=1,
            )
            attention = torch.ones(embeddings.shape[:2], dtype=torch.long, device=embeddings.device)
            with torch.inference_mode():
                output = backend.model(
                    inputs_embeds=embeddings,
                    attention_mask=attention,
                    use_cache=False,
                )
            return output.logits[0, -1]
        with Image.open(Path(prepared_input.image)) as source:
            pixel_image = source.convert("RGB")
        formatted = backend._format_prompt(prepared_input.prompt)  # noqa: SLF001
        inputs = backend.processor(images=pixel_image, text=formatted, return_tensors="pt")
        device = next(backend.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = backend.model(**inputs)
        return output.logits[0, -1]


class PrimaryScoreCache:
    """Content-addressed cache containing only the four registered likelihoods."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @staticmethod
    def key(*, prompt_sha256: str, image_sha256: str, model_sha256: str) -> str:
        payload = f"{PRIMARY_METHOD}|{prompt_sha256}|{image_sha256}|{model_sha256}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> PrimaryScore | None:
        path = self.root / key[:2] / f"{key}.json"
        if not path.is_file():
            return None
        row = json.loads(path.read_text(encoding="utf-8"))
        return PrimaryScore(
            selected_option_id=str(row["selected_option_id"]),
            choice_log_likelihoods={
                str(choice): float(value) for choice, value in row["choice_log_likelihoods"].items()
            },
            tied_maximum=bool(row["tied_maximum"]),
            tie_state=str(row["tie_state"]),
        )

    def put(self, key: str, result: PrimaryScore) -> None:
        path = self.root / key[:2] / f"{key}.json"
        if path.exists():
            existing = self.get(key)
            if existing != result:
                raise ValueError("primary score cache collision or scorer drift")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(result.to_dict(), handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()


def _token_ids(tokenizer: Any, value: str) -> list[int]:
    encoded = tokenizer(value, add_special_tokens=False)
    return [int(token) for token in encoded["input_ids"]]


__all__ = [
    "ChoiceLikelihoodBackend",
    "ImageFeatureCache",
    "LlavaPrimaryLikelihoodBackend",
    "PrimaryScore",
    "PrimaryScoreCache",
    "configure_frozen_tokenizer_runtime",
    "prepare_staged_gpu_image_features",
    "score_registered_options",
    "validate_tokenizer_contract",
]
