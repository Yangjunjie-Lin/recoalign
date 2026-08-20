"""Compatibility namespace for :mod:`recoalign.synthetic_world`."""

from recoalign.synthetic_world import GeneratorConfig, SyntheticWorldGenerator
from recoalign.synthetic_world.corruption import (
    CorruptionResult,
    flip_relation,
    randomize_graph,
    remove_relation,
    swap_entity,
)
from recoalign.synthetic_world.ontology import RELATIONS
from recoalign.synthetic_world.scene_graph import SceneGraph

__all__ = [
    "CorruptionResult",
    "GeneratorConfig",
    "RELATIONS",
    "SceneGraph",
    "SyntheticWorldGenerator",
    "flip_relation",
    "randomize_graph",
    "remove_relation",
    "swap_entity",
]
