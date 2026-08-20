"""Qwen-VL/Qwen2-VL adapter boundary; prompt semantics remain model-independent."""

from __future__ import annotations

from recoalign.models.vlm.boundary import AdapterBoundaryVLM


class QwenVLVLM(AdapterBoundaryVLM):
    model_id = "qwen_vl"


__all__ = ["QwenVLVLM"]
