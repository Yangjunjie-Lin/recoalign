"""Template-based questions derived from world state, never from an LLM."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from recoalign.synthetic_world.ontology import (
    COLORS,
    RELATIONS,
    SHAPES,
    SIZES,
    TEXTURES,
    normalize_relation,
)
from recoalign.synthetic_world.scene_graph import SceneGraph

QUESTION_TYPES = (
    "multi_hop",
    "relation_reasoning",
    "object_reasoning",
    "attribute_reasoning",
)

RELATION_PHRASES = {
    "left": "to the left of",
    "right": "to the right of",
    "above": "above",
    "below": "below",
    "front": "in front of",
    "behind": "behind",
    "near": "near",
    "far": "far from",
    "inside": "inside",
    "contains": "contains",
    "touching": "touching",
    "holding": "holding",
}


@dataclass(frozen=True)
class QuestionSpec:
    question: str
    answer: str
    choices: tuple[str, ...]
    question_type: str
    hop_depth: int
    query: dict[str, Any]


def object_description(node: dict[str, Any]) -> str:
    return (
        f"the {node['size']} {node['texture']} {node['color']} {node['shape']} "
        f"in the {node['category']} category"
    )


def object_list_from_world(
    objects: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> tuple[str, ...]:
    return tuple(object_description(node).removeprefix("the ") for node in objects)


def caption_from_world(
    objects: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    relations: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> str:
    """Losslessly verbalize the same declared facts carried by the oracle graph."""

    lookup = {str(node["id"]): object_description(node) for node in objects}
    inventory = "; ".join(object_list_from_world(objects))
    facts = "; ".join(
        f"{lookup[edge['subject']]} {_relation_predicate(edge['relation'])} "
        f"{lookup[edge['object']]}"
        for raw in relations
        for edge in (normalize_relation(dict(raw)),)
    )
    return f"Objects: {inventory}. Relations: {facts}."


def canonical_facts(
    objects: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    relations: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> tuple[str, ...]:
    object_facts = [
        "object|{id}|category={category}|shape={shape}|color={color}|size={size}|texture={texture}".format(
            **node
        )
        for node in sorted(objects, key=lambda item: str(item["id"]))
    ]
    relation_facts = [
        f"relation|{edge['subject']}|{edge['relation']}|{edge['object']}"
        for raw in relations
        for edge in (normalize_relation(dict(raw)),)
    ]
    return tuple((*object_facts, *relation_facts))


def facts_sha256(facts: tuple[str, ...] | list[str]) -> str:
    encoded = json.dumps(list(facts), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def generate_question(
    graph: SceneGraph,
    *,
    question_type: str,
    attribute: str = "color",
) -> QuestionSpec:
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"unknown question type: {question_type}")
    if not graph.edges:
        raise ValueError("question generation requires at least one relation")
    nodes = {str(node["id"]): node for node in graph.nodes}
    edges = [normalize_relation(edge) for edge in graph.edges]
    if question_type == "multi_hop":
        subject = edges[0]["subject"]
        target = edges[-1]["object"]
        answer = graph.relation(subject, target, max_hops=len(edges))
        if answer is None:
            raise ValueError("multi-hop path is not licensed by the scene-graph inference rules")
        question = (
            f"Following all {len(edges)} stated relations, where is "
            f"{object_description(nodes[subject])} relative to {object_description(nodes[target])}?"
        )
        return QuestionSpec(
            question,
            answer,
            RELATIONS,
            question_type,
            len(edges),
            {
                "subject": subject,
                "object": target,
                "answer_field": "relation",
                "supporting_edges": list(range(len(edges))),
            },
        )

    edge = edges[0]
    subject = edge["subject"]
    target = edge["object"]
    relation = edge["relation"]
    if question_type == "relation_reasoning":
        return QuestionSpec(
            (
                f"Where is {object_description(nodes[subject])} relative to "
                f"{object_description(nodes[target])}?"
            ),
            relation,
            RELATIONS,
            question_type,
            1,
            {
                "subject": subject,
                "object": target,
                "answer_field": "relation",
                "supporting_edges": [0],
            },
        )
    if question_type == "object_reasoning":
        return QuestionSpec(
            (
                f"What object shape {_relation_predicate(relation)} "
                f"{object_description(nodes[target])}?"
            ),
            str(nodes[subject]["shape"]),
            SHAPES,
            question_type,
            1,
            {
                "subject": subject,
                "object": target,
                "answer_field": "shape",
                "supporting_edges": [0],
            },
        )
    choices_by_attribute = {"color": COLORS, "size": SIZES, "texture": TEXTURES}
    if attribute not in choices_by_attribute:
        raise ValueError("attribute questions support color, size, or texture")
    return QuestionSpec(
        (
            f"What {attribute} is the object that {_relation_predicate(relation)} "
            f"{object_description(nodes[target])}?"
        ),
        str(nodes[subject][attribute]),
        choices_by_attribute[attribute],
        question_type,
        1,
        {
            "subject": subject,
            "object": target,
            "answer_field": attribute,
            "supporting_edges": [0],
        },
    )


def _relation_predicate(relation: str) -> str:
    phrase = RELATION_PHRASES[relation]
    return phrase if relation == "contains" else f"is {phrase}"


__all__ = [
    "QUESTION_TYPES",
    "QuestionSpec",
    "canonical_facts",
    "caption_from_world",
    "facts_sha256",
    "generate_question",
    "object_description",
    "object_list_from_world",
]
