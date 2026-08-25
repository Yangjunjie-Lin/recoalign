"""Exogenous, answer-role-neutral semantic scaffolds for PIVOT_EXP_A2."""

from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass
from typing import Any

from datasets.records import SceneRecord


@dataclass(frozen=True, order=True)
class EntityFact:
    alias: str
    shape: str
    color: str

    def canonical(self) -> str:
        return f"entity|{self.alias}|shape={self.shape}|color={self.color}"


@dataclass(frozen=True)
class SemanticScaffold:
    condition: str
    facts: tuple[EntityFact, ...]
    alias_by_object_id: dict[str, str]
    corruption_permutation: tuple[int, ...] | None

    @property
    def canonical_facts(self) -> tuple[str, ...]:
        return tuple(fact.canonical() for fact in sorted(self.facts))


def build_semantic_scaffolds(
    record: SceneRecord,
    *,
    seed: int,
    alias_salt: str,
    corruption_salt: str,
) -> tuple[SemanticScaffold, SemanticScaffold]:
    """Return oracle and fully false entity-attribute assignments.

    Alias assignment and the corruption derangement are bound to seed and scene ID but are
    independent of any model output.
    """

    objects = sorted((dict(node) for node in record.objects), key=lambda node: str(node["id"]))
    if len(objects) < 2:
        raise ValueError(f"{record.scene_id}: semantic rescue requires at least two entities")
    aliases = _aliases(len(objects))
    alias_rng = random.Random(_bound_seed(seed, record.scene_id, alias_salt))
    alias_rng.shuffle(aliases)
    alias_by_id = {
        str(node["id"]): alias for node, alias in zip(objects, aliases, strict=True)
    }
    oracle = tuple(
        EntityFact(
            alias=alias_by_id[str(node["id"])],
            shape=str(node["shape"]),
            color=str(node["color"]),
        )
        for node in objects
    )
    permutation = _false_derangement(
        tuple((fact.shape, fact.color) for fact in oracle),
        seed=_bound_seed(seed, record.scene_id, corruption_salt),
    )
    corrupted = tuple(
        EntityFact(alias=fact.alias, shape=oracle[source].shape, color=oracle[source].color)
        for fact, source in zip(oracle, permutation, strict=True)
    )
    _validate_scaffolds(record, oracle, corrupted, alias_by_id)
    return (
        SemanticScaffold(
            condition="oracle_semantics",
            facts=oracle,
            alias_by_object_id=alias_by_id,
            corruption_permutation=None,
        ),
        SemanticScaffold(
            condition="corrupted_semantics",
            facts=corrupted,
            alias_by_object_id=alias_by_id,
            corruption_permutation=permutation,
        ),
    )


def scaffold_by_condition(
    scaffolds: tuple[SemanticScaffold, SemanticScaffold], condition: str
) -> SemanticScaffold:
    for scaffold in scaffolds:
        if scaffold.condition == condition:
            return scaffold
    raise KeyError(condition)


def _false_derangement(
    assignments: tuple[tuple[str, str], ...], *, seed: int
) -> tuple[int, ...]:
    candidates = [
        permutation
        for permutation in itertools.permutations(range(len(assignments)))
        if all(
            assignments[source] != assignments[target]
            for target, source in enumerate(permutation)
        )
    ]
    if not candidates:
        raise ValueError("no fully false entity-attribute derangement exists")
    rng = random.Random(seed)
    return candidates[rng.randrange(len(candidates))]


def _validate_scaffolds(
    record: SceneRecord,
    oracle: tuple[EntityFact, ...],
    corrupted: tuple[EntityFact, ...],
    alias_by_id: dict[str, str],
) -> None:
    if len(oracle) != len(corrupted) or len(oracle) != len(alias_by_id):
        raise ValueError(f"{record.scene_id}: scaffold entity counts differ")
    if {fact.alias for fact in oracle} != {fact.alias for fact in corrupted}:
        raise ValueError(f"{record.scene_id}: corruption changed stable entity references")
    truth = {fact.alias: (fact.shape, fact.color) for fact in oracle}
    if any(truth[fact.alias] == (fact.shape, fact.color) for fact in corrupted):
        raise ValueError(f"{record.scene_id}: corrupted semantic binding remains true")
    forbidden = {"answer_object", "question_subject", "question_reference_object"}
    if forbidden & set(alias_by_id.values()):
        raise ValueError(f"{record.scene_id}: semantic aliases encode query roles")


def _aliases(count: int) -> list[str]:
    if count > 26:
        raise ValueError("semantic rescue supports at most 26 entities")
    return [f"Entity {chr(ord('A') + index)}" for index in range(count)]


def _bound_seed(seed: int, scene_id: str, salt: str) -> int:
    payload = f"{seed}|{scene_id}|{salt}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def semantic_metadata(scaffold: SemanticScaffold) -> dict[str, Any]:
    return {
        "condition": scaffold.condition,
        "alias_by_object_id": dict(sorted(scaffold.alias_by_object_id.items())),
        "corruption_permutation": (
            list(scaffold.corruption_permutation)
            if scaffold.corruption_permutation is not None
            else None
        ),
        "canonical_facts": list(scaffold.canonical_facts),
    }


__all__ = [
    "EntityFact",
    "SemanticScaffold",
    "build_semantic_scaffolds",
    "scaffold_by_condition",
    "semantic_metadata",
]
