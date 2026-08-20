"""Compatibility wrapper for the canonical ReCoAlign losses."""

from recoalign.models.recoalign.losses import (
    LOSS_NAMES,
    ReCoAlignLosses,
    reasoning_alignment_loss,
    semantic_preservation_loss,
    structural_consistency_loss,
)

__all__ = [
    "LOSS_NAMES",
    "ReCoAlignLosses",
    "reasoning_alignment_loss",
    "semantic_preservation_loss",
    "structural_consistency_loss",
]
