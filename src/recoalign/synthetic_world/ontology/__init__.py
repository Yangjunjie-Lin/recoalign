"""Finite, versioned semantic factors for the controlled synthetic world."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SHAPES = ("circle", "square", "triangle", "cube", "sphere")
CATEGORIES = ("animal", "vehicle", "object")
COLORS = ("red", "blue", "green", "yellow", "purple")
SIZES = ("small", "medium", "large")
TEXTURES = ("solid", "striped", "dotted")

SPATIAL_RELATIONS = ("left", "right", "above", "below")
DEPTH_RELATIONS = ("front", "behind")
DISTANCE_RELATIONS = ("near", "far")
CONTAINMENT_RELATIONS = ("inside", "contains")
INTERACTION_RELATIONS = ("touching", "holding")
RELATIONS = (
    *SPATIAL_RELATIONS,
    *DEPTH_RELATIONS,
    *DISTANCE_RELATIONS,
    *CONTAINMENT_RELATIONS,
    *INTERACTION_RELATIONS,
)

TRANSITIVE_RELATIONS = frozenset((*SPATIAL_RELATIONS, *DEPTH_RELATIONS, *CONTAINMENT_RELATIONS))
MULTIHOP_RELATIONS = (*SPATIAL_RELATIONS, *DEPTH_RELATIONS)
SYMMETRIC_RELATIONS = frozenset(("near", "far", "touching"))
INVERSE_RELATIONS = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
    "front": "behind",
    "behind": "front",
    "near": "near",
    "far": "far",
    "inside": "contains",
    "contains": "inside",
    "touching": "touching",
}


@dataclass(frozen=True)
class ObjectSpec:
    """One object sampled from independently controlled semantic factor vocabularies."""

    id: str
    category: str
    shape: str
    color: str
    size: str
    texture: str

    def __post_init__(self) -> None:
        validate_object(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "category": self.category,
            "shape": self.shape,
            "color": self.color,
            "size": self.size,
            "texture": self.texture,
        }


@dataclass(frozen=True)
class RelationSpec:
    """Canonical directed relation triple."""

    subject: str
    relation: str
    object: str

    def __post_init__(self) -> None:
        if not self.subject or not self.object or self.subject == self.object:
            raise ValueError("relation endpoints must be distinct non-empty object IDs")
        if self.relation not in RELATIONS:
            raise ValueError(f"unknown relation: {self.relation}")

    def to_dict(self) -> dict[str, str]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
        }


def validate_object(payload: dict[str, Any]) -> None:
    """Fail closed when an object escapes the declared ontology."""

    identifier = payload.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("object.id must be a non-empty string")
    vocabularies = {
        "category": CATEGORIES,
        "shape": SHAPES,
        "color": COLORS,
        "size": SIZES,
        "texture": TEXTURES,
    }
    for field, vocabulary in vocabularies.items():
        if payload.get(field) not in vocabulary:
            raise ValueError(f"object.{field} must be one of {vocabulary}")


def normalize_relation(payload: dict[str, Any]) -> dict[str, str]:
    """Accept legacy source/target edges and emit the canonical triple format."""

    relation = RelationSpec(
        subject=str(payload.get("subject", payload.get("source", ""))),
        relation=str(payload.get("relation", "")),
        object=str(payload.get("object", payload.get("target", ""))),
    )
    return relation.to_dict()


__all__ = [
    "CATEGORIES",
    "COLORS",
    "CONTAINMENT_RELATIONS",
    "DEPTH_RELATIONS",
    "DISTANCE_RELATIONS",
    "INTERACTION_RELATIONS",
    "INVERSE_RELATIONS",
    "MULTIHOP_RELATIONS",
    "ObjectSpec",
    "RELATIONS",
    "RelationSpec",
    "SHAPES",
    "SIZES",
    "SPATIAL_RELATIONS",
    "SYMMETRIC_RELATIONS",
    "TEXTURES",
    "TRANSITIVE_RELATIONS",
    "normalize_relation",
    "validate_object",
]
