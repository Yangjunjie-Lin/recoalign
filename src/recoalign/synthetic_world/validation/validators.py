"""Scientific-integrity validators for generated samples and datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema

from datasets.records import SceneRecord
from recoalign.synthetic_world.ontology import validate_object
from recoalign.synthetic_world.questions import (
    canonical_facts,
    caption_from_world,
    facts_sha256,
    object_list_from_world,
)
from recoalign.synthetic_world.scene_graph import SceneGraph
from recoalign.synthetic_world.splits import split_leakage_report


@lru_cache(maxsize=1)
def load_sample_schema() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[4] / "schemas" / "synthetic_scene.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _sample_validator() -> jsonschema.Draft202012Validator:
    schema = load_sample_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def validate_question(record: SceneRecord) -> dict[str, Any]:
    graph = SceneGraph(record.objects, record.relations)
    query = dict(record.metadata.get("query", {}))
    subject = str(query.get("subject", ""))
    target = str(query.get("object", ""))
    field = str(query.get("answer_field", ""))
    nodes = {str(node["id"]): node for node in record.objects}
    if subject not in nodes or target not in nodes:
        raise ValueError(f"{record.scene_id}: question query references an unknown object")
    hop_depth = int(record.metadata.get("hop_depth", 0))
    support = list(query.get("supporting_edges", []))
    if len(support) != hop_depth:
        raise ValueError(f"{record.scene_id}: supporting edge count does not equal hop depth")
    if any(not isinstance(index, int) or not 0 <= index < len(graph.edges) for index in support):
        raise ValueError(f"{record.scene_id}: supporting edge index is invalid")
    support_graph = SceneGraph(graph.nodes, tuple(graph.edges[index] for index in support))
    if field == "relation_sequence":
        sequence = tuple(str(edge["relation"]) for edge in support_graph.edges)
        declared = tuple(str(value) for value in query.get("relation_sequence", ()))
        if sequence != declared:
            raise ValueError(f"{record.scene_id}: declared relation sequence differs from support")
        expected = "+".join(sequence)
    else:
        derived_relation = support_graph.relation(subject, target, max_hops=hop_depth)
        if derived_relation != record.metadata.get("primary_relation"):
            raise ValueError(
                f"{record.scene_id}: supporting edges do not establish the query relation"
            )
        expected = derived_relation if field == "relation" else str(nodes[subject].get(field, ""))
    if expected != record.answer:
        raise ValueError(
            f"{record.scene_id}: answer {record.answer!r} does not match "
            f"derived answer {expected!r}"
        )
    if record.answer not in record.choices:
        raise ValueError(f"{record.scene_id}: answer is absent from choices")
    return {"valid": True, "derived_answer": expected, "hop_depth": hop_depth}


def validate_information_control(record: SceneRecord) -> dict[str, Any]:
    expected_caption = caption_from_world(record.objects, record.relations)
    expected_object_list = object_list_from_world(record.objects)
    facts = canonical_facts(record.objects, record.relations)
    digest = facts_sha256(facts)
    control = dict(record.metadata.get("information_control", {}))
    failures: list[str] = []
    if record.caption != expected_caption:
        failures.append("caption is not the deterministic lossless graph verbalization")
    if tuple(record.object_list) != expected_object_list:
        failures.append("object list is inconsistent with world objects")
    if tuple(record.metadata.get("canonical_facts", ())) != facts:
        failures.append("stored canonical facts differ from world state")
    if record.metadata.get("canonical_facts_sha256") != digest:
        failures.append("canonical_facts_sha256 mismatch")
    for key in ("graph_facts_sha256", "caption_facts_sha256"):
        if control.get(key) != digest:
            failures.append(f"information_control.{key} mismatch")
    if control.get("caption_graph_equivalent") is not True:
        failures.append("caption/graph equivalence flag is false")
    if failures:
        raise ValueError(f"{record.scene_id}: " + "; ".join(failures))
    return {"valid": True, "canonical_facts_sha256": digest}


def validate_sample(record: SceneRecord, *, check_image: bool = False) -> dict[str, Any]:
    payload = record.to_dict()
    _sample_validator().validate(payload)
    if payload["id"] != payload["scene_id"]:
        raise ValueError("id and scene_id compatibility alias differ")
    for node in record.objects:
        validate_object(node)
    graph = SceneGraph(record.objects, record.relations)
    if graph.to_dict() != payload["scene_graph"]:
        raise ValueError(f"{record.scene_id}: scene graph does not match record facts")
    world_payload = {
        "objects": record.objects,
        "relations": record.relations,
        "seed": record.metadata["sample_seed"],
    }
    world_digest = hashlib.sha256(
        json.dumps(world_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if world_digest != record.metadata.get("world_sha256"):
        raise ValueError(f"{record.scene_id}: world-state checksum mismatch")
    question = validate_question(record)
    control = validate_information_control(record)
    image_report: dict[str, Any] = {"checked": False}
    if check_image:
        if not record.image or not Path(record.image).is_file():
            raise ValueError(f"{record.scene_id}: materialized image does not exist")
        digest = hashlib.sha256(Path(record.image).read_bytes()).hexdigest()
        if digest != record.metadata.get("image_sha256"):
            raise ValueError(f"{record.scene_id}: image checksum mismatch")
        image_report = {"checked": True, "sha256": digest}
    return {
        "valid": True,
        "question": question,
        "information_control": control,
        "image": image_report,
    }


def validate_dataset(
    records: list[SceneRecord],
    *,
    split_strategy: str | None = None,
    check_images: bool = False,
) -> dict[str, Any]:
    identifiers = [record.scene_id for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("dataset contains duplicate sample IDs")
    for record in records:
        validate_sample(record, check_image=check_images)
    leakage = (
        split_leakage_report(records, split_strategy) if split_strategy is not None else None
    )
    if leakage is not None and not leakage["valid"]:
        raise ValueError("split leakage validation failed: " + "; ".join(leakage["violations"]))
    return {
        "valid": True,
        "count": len(records),
        "question_types": dict(Counter(str(row.metadata["question_type"]) for row in records)),
        "hop_depth": dict(Counter(str(row.metadata["hop_depth"]) for row in records)),
        "relations": dict(Counter(str(row.metadata["primary_relation"]) for row in records)),
        "leakage": leakage,
    }


__all__ = [
    "load_sample_schema",
    "validate_dataset",
    "validate_information_control",
    "validate_question",
    "validate_sample",
]
