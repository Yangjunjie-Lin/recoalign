"""Model adapters used by ReCoAlign.

The configuration and governance CLI must work in the CPU-only base install.
Torch-backed classes are therefore imported lazily when their public symbols are
actually accessed, instead of at package startup.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BaseVLM",
    "InternVLVLM",
    "Llava15VLM",
    "LlavaNextVLM",
    "OpenCLIPEncoder",
    "ReCoAlignConfig",
    "ReCoAlignModel",
    "QwenVLVLM",
    "ReferenceVLM",
]


_LAZY_IMPORTS = {
    "OpenCLIPEncoder": ("recoalign.models.openclip_encoder", "OpenCLIPEncoder"),
    "ReCoAlignConfig": ("recoalign.models.recoalign", "ReCoAlignConfig"),
    "ReCoAlignModel": ("recoalign.models.recoalign", "ReCoAlignModel"),
    "BaseVLM": ("recoalign.models.vlm", "BaseVLM"),
    "InternVLVLM": ("recoalign.models.vlm", "InternVLVLM"),
    "Llava15VLM": ("recoalign.models.vlm", "Llava15VLM"),
    "LlavaNextVLM": ("recoalign.models.vlm", "LlavaNextVLM"),
    "QwenVLVLM": ("recoalign.models.vlm", "QwenVLVLM"),
    "ReferenceVLM": ("recoalign.models.vlm", "ReferenceVLM"),
}


def __getattr__(name: str) -> Any:
    """Resolve an adapter only when its public symbol is accessed."""

    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
