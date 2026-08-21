"""Fact-equivalent JSON, triples, and secondary natural-language serializers."""

from __future__ import annotations

import hashlib
import json

from .relation_intervention import RelationFact
from .semantic_scaffold import EntityFact

_RELATION_PHRASES = {
    "left": "is left of",
    "right": "is right of",
    "above": "is above",
    "below": "is below",
    "front": "is in front of",
    "behind": "is behind",
    "near": "is near",
    "far": "is far from",
    "inside": "is inside",
    "contains": "contains",
    "touching": "is touching",
    "holding": "is holding",
}


def canonical_inventory(
    entities: tuple[EntityFact, ...], relations: tuple[RelationFact, ...]
) -> tuple[str, ...]:
    return tuple(
        sorted(
            [fact.canonical() for fact in entities]
            + [fact.canonical() for fact in relations]
        )
    )


def inventory_sha256(inventory: tuple[str, ...]) -> str:
    encoded = json.dumps(inventory, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def serialize_facts(
    entities: tuple[EntityFact, ...],
    relations: tuple[RelationFact, ...],
    serialization: str,
) -> str:
    entities = tuple(sorted(entities))
    relations = tuple(relations)
    if serialization == "canonical_json":
        payload = {
            "entities": {
                fact.alias: {"color": fact.color, "shape": fact.shape} for fact in entities
            },
            "relations": [
                {"object": fact.object, "predicate": fact.predicate, "subject": fact.subject}
                for fact in relations
            ],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if serialization == "canonical_triples":
        triples = [
            item
            for fact in entities
            for item in (
                f"({fact.alias},shape,{fact.shape})",
                f"({fact.alias},color,{fact.color})",
            )
        ]
        triples.extend(
            f"({fact.subject},{fact.predicate},{fact.object})" for fact in relations
        )
        return "Facts: " + " ".join(triples)
    if serialization == "natural_language":
        entity_text = "; ".join(
            f"{fact.alias} is a {fact.color} {fact.shape}" for fact in entities
        )
        relation_text = "; ".join(
            f"{fact.subject} {_RELATION_PHRASES[fact.predicate]} {fact.object}"
            for fact in relations
        )
        return f"Entities: {entity_text}. Relations: {relation_text}."
    raise ValueError(f"unknown causal-separation serialization: {serialization}")


__all__ = ["canonical_inventory", "inventory_sha256", "serialize_facts"]

