"""Unified, model-neutral VLM interfaces and registered backbone adapters."""

from .base import BaseVLM, PreparedInput, ReferenceVLM
from .evaluation import AnswerEvaluator, EvaluationResult
from .internvl import InternVLVLM
from .llava import Llava15VLM
from .llava_next import LlavaNextVLM
from .prompting import PromptProtocol, load_prompt_protocol
from .qwen_vl import QwenVLVLM
from .registry import ModelRegistry, create_vlm, get_model_registry

__all__ = [
    "AnswerEvaluator",
    "BaseVLM",
    "EvaluationResult",
    "InternVLVLM",
    "Llava15VLM",
    "LlavaNextVLM",
    "ModelRegistry",
    "PreparedInput",
    "PromptProtocol",
    "QwenVLVLM",
    "ReferenceVLM",
    "create_vlm",
    "get_model_registry",
    "load_prompt_protocol",
]
