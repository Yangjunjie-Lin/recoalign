"""InternVL adapter boundary; remote-code execution must be pinned before use."""

from __future__ import annotations

from recoalign.models.vlm.boundary import AdapterBoundaryVLM


class InternVLVLM(AdapterBoundaryVLM):
    model_id = "internvl"


__all__ = ["InternVLVLM"]
