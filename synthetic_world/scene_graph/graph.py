"""Compatibility import for the authoritative src/recoalign scene graph."""

from recoalign.synthetic_world.ontology import INVERSE_RELATIONS as OPPOSITE
from recoalign.synthetic_world.ontology import RELATIONS
from recoalign.synthetic_world.scene_graph.graph import SceneGraph

__all__ = ["OPPOSITE", "RELATIONS", "SceneGraph"]
