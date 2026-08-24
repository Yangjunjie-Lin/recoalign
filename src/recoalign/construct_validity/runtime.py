"""Frozen Transformers runtime compatibility for PIVOT_EXP_A3.

Transformers 4.52.4 defaults the slow Llama tokenizer to ``add_prefix_space=True`` even when the
caller explicitly uses ``add_special_tokens=False``. The preregistered A3 primary contract instead
pins candidates directly after ``FINAL_CHOICE=`` with no leading space. This module makes that
already-frozen boundary explicit for the one pinned local tokenizer; it does not alter prompts,
option IDs, token IDs, the checkpoint, or the parser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

FROZEN_TRANSFORMERS_VERSION = "4.52.4"
FROZEN_TOKENIZER = Path(__file__).resolve().parents[3] / "outputs/models/llava-v1.5-7b"
_PATCH_MARKER = "_recoalign_pivot_a3_no_prefix_space"


def configure_tokenizer_for_contract(tokenizer: Any) -> Any:
    """Set the preregistered no-leading-space behavior on a loaded tokenizer."""

    if not hasattr(tokenizer, "add_prefix_space"):
        raise TypeError("frozen PIVOT_EXP_A3 tokenizer must expose add_prefix_space")
    tokenizer.add_prefix_space = False
    return tokenizer


def configure_frozen_tokenizer_runtime() -> dict[str, Any]:
    """Patch AutoTokenizer loading only for the pinned local LLaVA tokenizer."""

    import transformers
    from transformers import AutoTokenizer

    if transformers.__version__ != FROZEN_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "PIVOT_EXP_A3 requires frozen Transformers "
            f"{FROZEN_TRANSFORMERS_VERSION}, observed {transformers.__version__}"
        )
    descriptor = AutoTokenizer.__dict__["from_pretrained"]
    function = descriptor.__func__
    if getattr(function, _PATCH_MARKER, False):
        return {
            "configured": True,
            "already_configured": True,
            "transformers_version": transformers.__version__,
            "tokenizer_path": str(FROZEN_TOKENIZER),
            "add_prefix_space": False,
        }

    original = function

    def frozen_from_pretrained(
        cls: type[Any], pretrained_model_name_or_path: str | Path, *inputs: Any, **kwargs: Any
    ) -> Any:
        candidate = Path(str(pretrained_model_name_or_path))
        try:
            is_frozen = candidate.resolve() == FROZEN_TOKENIZER.resolve()
        except OSError:
            is_frozen = False
        if is_frozen:
            kwargs["add_prefix_space"] = False
        tokenizer = original(cls, pretrained_model_name_or_path, *inputs, **kwargs)
        return configure_tokenizer_for_contract(tokenizer) if is_frozen else tokenizer

    setattr(frozen_from_pretrained, _PATCH_MARKER, True)
    AutoTokenizer.from_pretrained = classmethod(frozen_from_pretrained)
    return {
        "configured": True,
        "already_configured": False,
        "transformers_version": transformers.__version__,
        "tokenizer_path": str(FROZEN_TOKENIZER),
        "add_prefix_space": False,
    }


__all__ = ["configure_frozen_tokenizer_runtime", "configure_tokenizer_for_contract"]
