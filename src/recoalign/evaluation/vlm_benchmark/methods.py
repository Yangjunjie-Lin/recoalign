"""Method-neutral prompt construction and strict ReCoAlign dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recoalign.models.vlm.base import BaseVLM

from .adapters import BenchmarkSample

METHODS = (
    "original_vlm",
    "caption_reasoning",
    "oracle_graph_prompt",
    "recoalign",
    "no_structure_token",
    "random_structure_token",
    "fixed_graph_encoder",
    "no_structural_loss",
    "no_semantic_loss",
    "no_reasoning_alignment",
    "synthetic_only",
    "real_only",
    "mixed_training",
)
INTERFACE_METHODS = {
    "recoalign",
    "no_structure_token",
    "random_structure_token",
    "fixed_graph_encoder",
    "no_structural_loss",
    "no_semantic_loss",
    "no_reasoning_alignment",
    "synthetic_only",
    "real_only",
    "mixed_training",
}


@dataclass(frozen=True)
class MethodReadiness:
    method: str
    ready: bool
    reason: str | None
    checkpoint: str | None
    oracle_graph_at_inference: bool


def validate_method(method: str) -> str:
    if method not in METHODS:
        raise ValueError(f"unsupported evaluation method {method!r}")
    return method


def method_readiness(
    method: str,
    config: dict[str, Any],
    *,
    sample: BenchmarkSample | None = None,
    model: BaseVLM | None = None,
) -> MethodReadiness:
    validate_method(method)
    method_config = config.get("method", {})
    if not isinstance(method_config, dict):
        raise ValueError("method configuration must be a mapping")
    if method == "caption_reasoning" and sample is not None and "caption" not in sample.context:
        return MethodReadiness(method, False, "caption context is absent", None, False)
    if (
        method == "oracle_graph_prompt"
        and sample is not None
        and "oracle_graph" not in sample.context
    ):
        return MethodReadiness(method, False, "oracle graph annotation is absent", None, True)
    if method in INTERFACE_METHODS:
        checkpoint = _interface_checkpoint(method_config, config.get("seed"))
        if checkpoint is None:
            return MethodReadiness(
                method, False, "interface checkpoint is not configured", None, False
            )
        if not Path(checkpoint).is_file():
            return MethodReadiness(
                method, False, "interface checkpoint does not exist", checkpoint, False
            )
        if model is not None and not model.supports_interface_injection:
            return MethodReadiness(
                method,
                False,
                "backbone runtime does not implement hidden-context interface injection",
                checkpoint,
                False,
            )
        return MethodReadiness(method, True, None, checkpoint, False)
    return MethodReadiness(
        method,
        True,
        None,
        None,
        method == "oracle_graph_prompt",
    )


def render_prompt(sample: BenchmarkSample, method: str, protocol: dict[str, Any]) -> str:
    validate_method(method)
    instruction = str(
        protocol.get(
            "instruction",
            "Answer the visual question using only the supplied image and permitted context.",
        )
    ).strip()
    lines = [instruction]
    if method == "caption_reasoning":
        caption = sample.context.get("caption")
        if caption is None:
            raise ValueError(f"{sample.sample_id} has no controlled caption context")
        lines.append(f"Auxiliary caption: {caption}")
    elif method == "oracle_graph_prompt":
        graph = sample.context.get("oracle_graph")
        if graph is None:
            raise ValueError(f"{sample.sample_id} has no oracle graph context")
        lines.append(f"Oracle scene graph (upper-bound baseline only): {_serialize(graph)}")
    lines.append(f"Question: {sample.question}")
    if sample.choices:
        lines.append("Options:")
        for label, text in zip(sample.choices, sample.choice_texts, strict=True):
            lines.append(f"[{label}] {text}")
        lines.append("Return exactly one option label.")
    else:
        lines.append("Return only the short answer.")
    return "\n".join(lines)


def generate_prediction(
    model: BaseVLM,
    sample: BenchmarkSample,
    method: str,
    config: dict[str, Any],
) -> str:
    prompt = render_prompt(sample, method, dict(config.get("prompt", {})))
    generation = dict(config.get("generation", {}))
    if method in INTERFACE_METHODS:
        readiness = method_readiness(method, config, sample=sample, model=model)
        if not readiness.ready or readiness.checkpoint is None:
            raise RuntimeError(readiness.reason or "ReCoAlign method is not ready")
        interface_config = dict(config.get("method", {}))
        interface_config["evaluation_variant"] = method
        return model.generate_with_interface(
            prompt,
            image=sample.image,
            interface_checkpoint=readiness.checkpoint,
            interface_config=interface_config,
            **generation,
        )
    return model.generate(prompt, image=sample.image, **generation)


def _interface_checkpoint(method: dict[str, Any], seed: Any) -> str | None:
    by_seed = method.get("checkpoint_by_seed", {})
    if isinstance(by_seed, dict) and str(seed) in by_seed:
        return str(by_seed[str(seed)])
    value = method.get("interface_checkpoint")
    return str(value) if value else None


def _serialize(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "; ".join(f"{key}={_serialize(item)}" for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return " | ".join(_serialize(item) for item in value)
    return str(value)


__all__ = [
    "INTERFACE_METHODS",
    "METHODS",
    "MethodReadiness",
    "generate_prediction",
    "method_readiness",
    "render_prompt",
    "validate_method",
]
