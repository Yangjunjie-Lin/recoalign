"""Compatibility exports for the authoritative corruption implementation."""

from recoalign.synthetic_world.corruption import (
    SEMANTIC_FLIPS,
    CorruptionResult,
    flip_relation,
    randomize_graph,
    remove_relation,
    swap_entity,
)

__all__ = [
    "CorruptionResult",
    "SEMANTIC_FLIPS",
    "flip_relation",
    "randomize_graph",
    "remove_relation",
    "swap_entity",
]
