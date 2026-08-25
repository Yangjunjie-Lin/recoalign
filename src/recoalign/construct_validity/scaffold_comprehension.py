"""Relation-free semantic scaffold construction for PIVOT_EXP_A3."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

from datasets.records import SceneRecord

MANIPULATIONS = ("M0", "M1", "M2")
EVIDENCE_TRUTH = ("oracle", "corrupted")
_ALIASES = {
    "M0": ("Entity A", "Entity B", "Entity C", "Entity D"),
    "M1": ("E1", "E2", "E3", "E4"),
    "M2": ("E1", "E2", "E3", "E4"),
}


@dataclass(frozen=True, order=True)
class EntityBinding:
    entity_id: str
    shape: str
    color: str
    source_object_id: str

    @property
    def description(self) -> str:
        return f"{self.color} {self.shape}"

    def canonical(self) -> str:
        return f"entity|{self.entity_id}|shape={self.shape}|color={self.color}"


@dataclass(frozen=True)
class ConstructScaffold:
    scene_id: str
    manipulation: str
    evidence_truth: str
    bindings: tuple[EntityBinding, ...]
    row_order: tuple[str, ...]
    alias_by_object_id: dict[str, str]
    corruption_shift: int | None

    @property
    def target_entity_id(self) -> str:
        return _ALIASES[self.manipulation][1]

    @property
    def canonical_facts(self) -> tuple[str, ...]:
        return tuple(sorted(binding.canonical() for binding in self.bindings))

    def by_entity(self) -> dict[str, EntityBinding]:
        return {binding.entity_id: binding for binding in self.bindings}


def build_scaffold_pair(
    record: SceneRecord,
    manipulation: str,
    *,
    seed: int,
    alias_salt: str = "pivot-exp-a3-alias-assignment-v1",
    order_salt: str = "pivot-exp-a3-table-order-v1",
    corruption_salt: str = "pivot-exp-a3-binding-corruption-v1",
) -> tuple[ConstructScaffold, ConstructScaffold]:
    if manipulation not in MANIPULATIONS:
        raise ValueError(f"unregistered semantic manipulation: {manipulation}")
    objects = sorted((dict(node) for node in record.objects), key=lambda node: str(node["id"]))
    if len(objects) != 4:
        raise ValueError(f"{record.scene_id}: PIVOT_EXP_A3 requires exactly four objects")
    aliases = list(_ALIASES[manipulation])
    random.Random(_bound_seed(seed, record.scene_id, alias_salt, manipulation)).shuffle(aliases)
    alias_by_object = {
        str(node["id"]): alias for node, alias in zip(objects, aliases, strict=True)
    }
    oracle_by_alias = {
        alias_by_object[str(node["id"])]: EntityBinding(
            entity_id=alias_by_object[str(node["id"])],
            shape=str(node["shape"]),
            color=str(node["color"]),
            source_object_id=str(node["id"]),
        )
        for node in objects
    }
    registered_aliases = list(_ALIASES[manipulation])
    row_order = list(registered_aliases)
    random.Random(_bound_seed(seed, record.scene_id, order_salt, manipulation)).shuffle(row_order)
    shift = 1 + _bound_seed(seed, record.scene_id, corruption_salt, manipulation) % 3
    corrupted_by_alias = {}
    for index, alias in enumerate(registered_aliases):
        source = oracle_by_alias[registered_aliases[(index + shift) % 4]]
        corrupted_by_alias[alias] = EntityBinding(
            entity_id=alias,
            shape=source.shape,
            color=source.color,
            source_object_id=source.source_object_id,
        )
    oracle = ConstructScaffold(
        scene_id=record.scene_id,
        manipulation=manipulation,
        evidence_truth="oracle",
        bindings=tuple(oracle_by_alias[alias] for alias in registered_aliases),
        row_order=tuple(row_order),
        alias_by_object_id=alias_by_object,
        corruption_shift=None,
    )
    corrupted = ConstructScaffold(
        scene_id=record.scene_id,
        manipulation=manipulation,
        evidence_truth="corrupted",
        bindings=tuple(corrupted_by_alias[alias] for alias in registered_aliases),
        row_order=tuple(row_order),
        alias_by_object_id=alias_by_object,
        corruption_shift=int(shift),
    )
    validate_scaffold_pair(oracle, corrupted)
    return oracle, corrupted


def serialize_scaffold(scaffold: ConstructScaffold) -> str:
    by_entity = scaffold.by_entity()
    if scaffold.manipulation == "M0":
        payload = {
            "entities": {
                entity: {"color": by_entity[entity].color, "shape": by_entity[entity].shape}
                for entity in scaffold.row_order
            },
            "relations": [],
        }
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    if scaffold.manipulation == "M1":
        rows = ["ENTITY_TABLE"]
        rows.extend(
            f"{entity} | shape={by_entity[entity].shape} | color={by_entity[entity].color}"
            for entity in scaffold.row_order
        )
        return "\n".join(rows)
    if scaffold.manipulation == "M2":
        return "VISUAL_OBJECT_LEGEND\nRead the four registered legend rows in the image."
    raise ValueError(scaffold.manipulation)


def validate_scaffold_pair(
    oracle: ConstructScaffold, corrupted: ConstructScaffold
) -> dict[str, bool]:
    oracle_map = oracle.by_entity()
    corrupted_map = corrupted.by_entity()
    assertions = {
        "same_scene": oracle.scene_id == corrupted.scene_id,
        "same_manipulation": oracle.manipulation == corrupted.manipulation,
        "truth_labels": oracle.evidence_truth == "oracle"
        and corrupted.evidence_truth == "corrupted",
        "four_entities": len(oracle.bindings) == len(corrupted.bindings) == 4,
        "same_entity_ids": set(oracle_map) == set(corrupted_map),
        "same_row_order": oracle.row_order == corrupted.row_order,
        "same_attribute_multiset": sorted(
            (value.shape, value.color, value.source_object_id) for value in oracle.bindings
        )
        == sorted(
            (value.shape, value.color, value.source_object_id) for value in corrupted.bindings
        ),
        "all_corrupted_bindings_false": all(
            (oracle_map[entity].shape, oracle_map[entity].color)
            != (corrupted_map[entity].shape, corrupted_map[entity].color)
            for entity in oracle_map
        ),
        "alias_mapping_stable": oracle.alias_by_object_id == corrupted.alias_by_object_id,
    }
    if not all(assertions.values()):
        failed = sorted(name for name, value in assertions.items() if not value)
        raise ValueError(f"invalid oracle/corrupted scaffold pair: {failed}")
    return assertions


def scaffold_metadata(scaffold: ConstructScaffold) -> dict[str, Any]:
    return {
        "scene_id": scaffold.scene_id,
        "manipulation": scaffold.manipulation,
        "evidence_truth": scaffold.evidence_truth,
        "row_order": list(scaffold.row_order),
        "alias_by_object_id": dict(sorted(scaffold.alias_by_object_id.items())),
        "corruption_shift": scaffold.corruption_shift,
        "canonical_facts": list(scaffold.canonical_facts),
        "canonical_facts_sha256": hashlib.sha256(
            json.dumps(scaffold.canonical_facts, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _bound_seed(seed: int, scene_id: str, salt: str, manipulation: str) -> int:
    payload = f"{seed}|{scene_id}|{salt}|{manipulation}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


__all__ = [
    "ConstructScaffold",
    "EVIDENCE_TRUTH",
    "EntityBinding",
    "MANIPULATIONS",
    "build_scaffold_pair",
    "scaffold_metadata",
    "serialize_scaffold",
    "validate_scaffold_pair",
]
