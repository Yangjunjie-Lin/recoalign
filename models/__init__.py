"""Compatibility model namespace with lazy optional-dependency imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["ReCoAlignConfig", "ReCoAlignModel"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("recoalign.models.recoalign"), name)
    globals()[name] = value
    return value
