"""Information-controlled evidence views for EXP001 and graph ablations."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from datasets.records import SceneRecord
from recoalign.synthetic_world.corruption import randomize_graph
from recoalign.synthetic_world.ontology import normalize_relation
from recoalign.synthetic_world.questions.generation import caption_from_world, facts_sha256
from recoalign.synthetic_world.scene_graph import SceneGraph


class Condition(str, Enum):
    IMAGE = "image_only"
    OBJECT_LIST = "object_list"
    CAPTION = "caption"
    SCENE_GRAPH = "scene_graph"
    GRAPH_ORDER_SHUFFLED = "graph_order_shuffled"
    TEXT_UNORDERED = "text_unordered"
    GRAPH_SERIALIZATION_TRIPLES = "graph_serialization_triples"
    GRAPH_SERIALIZATION_JSON = "graph_serialization_json"
    PARTIAL_GRAPH = "partial_graph"
    RANDOM_GRAPH = "random_graph"
    CORRUPTED_GRAPH = "corrupted_graph"


class InputSetting(str, Enum):
    NATURAL = "natural"
    TOKEN_MATCHED = "token_matched"


PRIMARY_CONDITIONS = (
    Condition.IMAGE.value,
    Condition.OBJECT_LIST.value,
    Condition.CAPTION.value,
    Condition.SCENE_GRAPH.value,
)
ABLATION_CONDITIONS = (
    Condition.GRAPH_ORDER_SHUFFLED.value,
    Condition.TEXT_UNORDERED.value,
    Condition.GRAPH_SERIALIZATION_TRIPLES.value,
    Condition.GRAPH_SERIALIZATION_JSON.value,
)
CONDITIONS = tuple(condition.value for condition in Condition)
TOKEN_CONTROLLED_CONDITIONS = {Condition.CAPTION.value, Condition.SCENE_GRAPH.value}


@dataclass(frozen=True)
class EvidenceView:
    """One auditable representation of the evidence supplied to a model."""

    condition: str
    text: str
    semantic_facts_sha256: str | None
    object_count: int
    attribute_count: int
    relation_count: int
    semantic_units: int
    serialization: str
    padding_units: int = 0
    natural_tokens: int | None = None
    matched_tokens: int | None = None
    token_match_delta: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "semantic_facts_sha256": self.semantic_facts_sha256,
            "object_count": self.object_count,
            "attribute_count": self.attribute_count,
            "relation_count": self.relation_count,
            "semantic_units": self.semantic_units,
            "serialization": self.serialization,
            "padding_units": self.padding_units,
            "natural_tokens": self.natural_tokens,
            "matched_tokens": self.matched_tokens,
            "token_match_delta": self.token_match_delta,
        }


def graph_variant(record: SceneRecord, condition: str) -> SceneGraph:
    graph = SceneGraph(record.objects, record.relations)
    if condition == Condition.PARTIAL_GRAPH:
        return graph.partial()
    if condition == Condition.RANDOM_GRAPH:
        seed = int(record.metadata.get("sample_seed", record.metadata.get("seed", 0)))
        return randomize_graph(graph, seed=seed + 1).corrupted
    if condition == Condition.CORRUPTED_GRAPH:
        seed = int(record.metadata.get("sample_seed", record.metadata.get("seed", 0)))
        return graph.corrupted(seed)
    return graph


def caption(record: SceneRecord) -> str:
    return record.caption or caption_from_world(record.objects, record.relations)


def lexical_token_count(text: str) -> int:
    """Dependency-free reference counter used only when no model tokenizer exists."""

    return len(re.findall(r"<pad>|[A-Za-z0-9_]+|[^\w\s]", text, flags=re.UNICODE))


def evidence_view(record: SceneRecord, condition: str) -> EvidenceView:
    """Serialize one condition while retaining declared semantic-unit accounting."""

    value = str(condition)
    graph = graph_variant(record, value)
    full_graph_family = value in {
        Condition.CAPTION.value,
        Condition.SCENE_GRAPH.value,
        Condition.GRAPH_ORDER_SHUFFLED.value,
        Condition.TEXT_UNORDERED.value,
        Condition.GRAPH_SERIALIZATION_TRIPLES.value,
        Condition.GRAPH_SERIALIZATION_JSON.value,
    }
    graph_family = value in {
        Condition.SCENE_GRAPH.value,
        Condition.GRAPH_ORDER_SHUFFLED.value,
        Condition.GRAPH_SERIALIZATION_TRIPLES.value,
        Condition.GRAPH_SERIALIZATION_JSON.value,
        Condition.PARTIAL_GRAPH.value,
        Condition.RANDOM_GRAPH.value,
        Condition.CORRUPTED_GRAPH.value,
    }
    if value == Condition.IMAGE.value:
        text = "Evidence: image only."
        object_count = attribute_count = relation_count = 0
        digest = None
        serialization = "image"
    elif value == Condition.OBJECT_LIST.value:
        values = record.object_list or tuple(
            " ".join(
                str(node.get(key, "unknown"))
                for key in ("size", "texture", "color", "shape", "category")
            )
            for node in record.objects
        )
        text = f"Evidence: unordered object list: {', '.join(values)}."
        object_count = len(record.objects)
        attribute_count = _attribute_count(record.objects)
        relation_count = 0
        digest = facts_sha256(_object_facts(record))
        serialization = "object_list"
    elif value == Condition.CAPTION.value:
        text = f"Evidence: caption: {caption(record)}"
        object_count = len(record.objects)
        attribute_count = _attribute_count(record.objects)
        relation_count = len(record.relations)
        digest = _declared_fact_hash(record)
        serialization = "natural_language"
    elif value == Condition.TEXT_UNORDERED.value:
        text = f"Evidence: unordered text: {_unordered_text(record)}"
        object_count = len(record.objects)
        attribute_count = _attribute_count(record.objects)
        relation_count = len(record.relations)
        digest = _declared_fact_hash(record)
        serialization = "unordered_text"
    elif graph_family:
        edges = list(graph.edges)
        serialization = "canonical"
        if value == Condition.GRAPH_ORDER_SHUFFLED.value:
            rng = random.Random(int(record.metadata.get("sample_seed", 0)) + 1009)
            rng.shuffle(edges)
            if len(edges) > 1 and edges == list(graph.edges):
                edges.reverse()
            serialization = "canonical_shuffled_edges"
        elif value == Condition.GRAPH_SERIALIZATION_TRIPLES.value:
            serialization = "tuples"
        elif value == Condition.GRAPH_SERIALIZATION_JSON.value:
            serialization = "json"
        graph = SceneGraph(graph.nodes, tuple(edges))
        text = f"Evidence: scene graph ({serialization}).\n{_serialize_graph(graph, serialization)}"
        object_count = len(graph.nodes)
        attribute_count = _attribute_count(graph.nodes)
        relation_count = len(graph.edges)
        digest = _declared_fact_hash(record) if full_graph_family else None
    else:
        raise ValueError(f"unknown evidence condition: {condition}")
    return EvidenceView(
        condition=value,
        text=text,
        semantic_facts_sha256=digest,
        object_count=object_count,
        attribute_count=attribute_count,
        relation_count=relation_count,
        semantic_units=object_count + attribute_count + relation_count,
        serialization=serialization,
    )


def controlled_evidence_view(
    record: SceneRecord,
    condition: str,
    *,
    setting: str = InputSetting.NATURAL.value,
    count_tokens: Callable[[str], int] = lexical_token_count,
    tolerance: int = 1,
) -> EvidenceView:
    """Return natural or token-matched Caption/Graph evidence for the same scene facts."""

    value = str(condition)
    natural = evidence_view(record, value)
    natural_count = count_tokens(natural.text)
    natural = replace(
        natural,
        natural_tokens=natural_count,
        matched_tokens=natural_count,
        token_match_delta=0,
    )
    if setting == InputSetting.NATURAL.value or value not in TOKEN_CONTROLLED_CONDITIONS:
        return natural
    if setting != InputSetting.TOKEN_MATCHED.value:
        raise ValueError(f"unknown input setting: {setting}")
    caption_view = evidence_view(record, Condition.CAPTION.value)
    graph_view = evidence_view(record, Condition.SCENE_GRAPH.value)
    left, right = match_token_budget(
        caption_view,
        graph_view,
        count_tokens=count_tokens,
        tolerance=tolerance,
    )
    return left if value == Condition.CAPTION.value else right


def match_token_budget(
    caption_view: EvidenceView,
    graph_view: EvidenceView,
    *,
    count_tokens: Callable[[str], int] = lexical_token_count,
    tolerance: int = 1,
) -> tuple[EvidenceView, EvidenceView]:
    """Pad the shorter view with content-free sentinels until token counts are matched."""

    if tolerance < 0:
        raise ValueError("token-match tolerance must be non-negative")
    caption_tokens = count_tokens(caption_view.text)
    graph_tokens = count_tokens(graph_view.text)
    caption_text = caption_view.text
    graph_text = graph_view.text
    caption_padding = graph_padding = 0
    maximum_steps = max(64, abs(caption_tokens - graph_tokens) * 8 + 32)
    for _ in range(maximum_steps):
        delta = caption_tokens - graph_tokens
        if abs(delta) <= tolerance:
            break
        if delta < 0:
            caption_text += " <pad>"
            caption_padding += 1
            updated = count_tokens(caption_text)
            if updated <= caption_tokens:
                raise ValueError("model token counter did not advance after adding a pad sentinel")
            caption_tokens = updated
        else:
            graph_text += " <pad>"
            graph_padding += 1
            updated = count_tokens(graph_text)
            if updated <= graph_tokens:
                raise ValueError("model token counter did not advance after adding a pad sentinel")
            graph_tokens = updated
    delta = abs(caption_tokens - graph_tokens)
    if delta > tolerance:
        raise ValueError(
            f"could not token-match caption and graph within tolerance {tolerance}; delta={delta}"
        )
    return (
        replace(
            caption_view,
            text=caption_text,
            padding_units=caption_padding,
            natural_tokens=count_tokens(caption_view.text),
            matched_tokens=caption_tokens,
            token_match_delta=delta,
        ),
        replace(
            graph_view,
            text=graph_text,
            padding_units=graph_padding,
            natural_tokens=count_tokens(graph_view.text),
            matched_tokens=graph_tokens,
            token_match_delta=delta,
        ),
    )


def condition_text(
    record: SceneRecord,
    condition: str,
    *,
    setting: str = InputSetting.NATURAL.value,
    count_tokens: Callable[[str], int] = lexical_token_count,
    token_match_tolerance: int = 1,
) -> str:
    question = (
        f"Question: {record.question}\n"
        f"Answer with exactly one option: {', '.join(record.choices)}."
    )
    evidence = controlled_evidence_view(
        record,
        condition,
        setting=setting,
        count_tokens=lambda text: count_tokens(f"{text}\n{question}"),
        tolerance=token_match_tolerance,
    )
    return f"{evidence.text}\n{question}"


def condition_image(record: SceneRecord, condition: str) -> str | None:
    if str(condition) in CONDITIONS:
        return record.image or None
    raise ValueError(f"unknown evidence condition: {condition}")


def _serialize_graph(graph: SceneGraph, serialization: str) -> str:
    if serialization in {"canonical", "canonical_shuffled_edges"}:
        return graph.text()
    if serialization == "tuples":
        nodes = " ".join(
            "({id},{category},{shape},{color},{size},{texture})".format(**node)
            for node in graph.nodes
        )
        edges = " ".join(
            f"({edge['subject']},{edge['relation']},{edge['object']})"
            for raw in graph.edges
            for edge in (normalize_relation(dict(raw)),)
        )
        return f"Objects: {nodes}\nRelations: {edges}"
    if serialization == "json":
        return json.dumps(graph.to_dict(), sort_keys=True, separators=(",", ":"))
    raise ValueError(f"unknown graph serialization: {serialization}")


def _unordered_text(record: SceneRecord) -> str:
    tokens: list[str] = []
    for node in record.objects:
        tokens.extend(
            str(node[key])
            for key in ("id", "category", "shape", "color", "size", "texture")
        )
    for raw in record.relations:
        edge = normalize_relation(dict(raw))
        tokens.extend((edge["subject"], edge["relation"], edge["object"]))
    random.Random(int(record.metadata.get("sample_seed", 0)) + 2027).shuffle(tokens)
    return " ".join(tokens)


def _attribute_count(objects: Any) -> int:
    return sum(
        key in node
        for node in objects
        for key in ("category", "shape", "color", "size", "texture")
    )


def _object_facts(record: SceneRecord) -> tuple[str, ...]:
    return tuple(
        "object|{id}|category={category}|shape={shape}|color={color}|size={size}|texture={texture}".format(
            **node
        )
        for node in sorted(record.objects, key=lambda item: str(item["id"]))
    )


def _declared_fact_hash(record: SceneRecord) -> str:
    control = record.metadata.get("information_control", {})
    digest = control.get("caption_facts_sha256")
    graph_digest = control.get("graph_facts_sha256")
    if not digest or digest != graph_digest:
        raise ValueError(f"{record.scene_id}: caption/graph semantic fact hashes do not match")
    return str(digest)


__all__ = [
    "ABLATION_CONDITIONS",
    "CONDITIONS",
    "PRIMARY_CONDITIONS",
    "Condition",
    "EvidenceView",
    "InputSetting",
    "caption",
    "condition_image",
    "condition_text",
    "controlled_evidence_view",
    "evidence_view",
    "graph_variant",
    "lexical_token_count",
    "match_token_budget",
]
