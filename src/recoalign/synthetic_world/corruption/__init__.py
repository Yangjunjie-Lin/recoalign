"""Controlled graph-corruption framework used by structural-necessity experiments."""

from .operators import (
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
