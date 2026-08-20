"""Oracle/metadata scene-graph encoder used for mechanism validation."""

from __future__ import annotations

from datasets.records import SceneRecord
from models.reasoning_interface.interfaces import StructuredRepresentation, VisualRepresentation


class SceneGraphEncoder:
    """Convert a scene record into an explicit intermediate representation."""

    def encode(self, visual: VisualRepresentation) -> StructuredRepresentation:
        payload = visual.values
        if not isinstance(payload, SceneRecord):
            raise TypeError("SceneGraphEncoder expects a SceneRecord payload in Phase 1")
        return StructuredRepresentation(
            nodes=payload.objects,
            edges=payload.relations,
            source="synthetic_world.oracle_graph",
            confidence=1.0,
            metadata={"scene_id": payload.scene_id},
        )
