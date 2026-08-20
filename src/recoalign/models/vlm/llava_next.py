"""LLaVA-NeXT adapter boundary; checkpoint execution is enabled after local pinning."""

from __future__ import annotations

from recoalign.models.vlm.boundary import AdapterBoundaryVLM


class LlavaNextVLM(AdapterBoundaryVLM):
    model_id = "llava_next"


__all__ = ["LlavaNextVLM"]
