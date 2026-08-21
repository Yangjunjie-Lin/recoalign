"""Count-preserving true and false relation interventions for PIVOT_EXP_A2."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from datasets.records import SceneRecord
from recoalign.synthetic_world.ontology import RELATIONS, normalize_relation

_INVERSE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
    "front": "behind",
    "behind": "front",
    "inside": "contains",
    "contains": "inside",
    "near": "near",
    "far": "far",
    "touching": "touching",
    "holding": "inside",
}
_PREFERRED_FALSE = {
    "left": ("right", "above", "far"),
    "right": ("left", "below", "far"),
    "above": ("below", "right", "far"),
    "below": ("above", "left", "far"),
    "front": ("behind", "far", "left"),
    "behind": ("front", "far", "right"),
    "near": ("far", "left", "above"),
    "far": ("near", "right", "below"),
    "inside": ("contains", "far", "left"),
    "contains": ("inside", "far", "right"),
    "touching": ("far", "above", "left"),
    "holding": ("far", "below", "right"),
}


@dataclass(frozen=True, order=True)
class RelationFact:
    subject: str
    predicate: str
    object: str

    def canonical(self) -> str:
        return f"relation|{self.subject}|{self.predicate}|{self.object}"


@dataclass(frozen=True)
class RelationIntervention:
    condition: str
    facts: tuple[RelationFact, ...]
    source_edge_indices: tuple[int, ...]
    corruption_methods: tuple[str, ...]

    @property
    def canonical_facts(self) -> tuple[str, ...]:
        return tuple(fact.canonical() for fact in self.facts)


def build_relation_interventions(
    record: SceneRecord,
    *,
    alias_by_object_id: dict[str, str],
    seed: int,
    corruption_salt: str,
) -> tuple[RelationIntervention, RelationIntervention]:
    query = dict(record.metadata.get("query", {}))
    indices = tuple(int(index) for index in query.get("supporting_edges", ()))
    if not indices:
        raise ValueError(f"{record.scene_id}: query has no registered supporting edges")
    if str(query.get("answer_field")) == "relation" and len(indices) == 1:
        raise ValueError(
            f"{record.scene_id}: direct relation-answer trial is ineligible for the main matrix"
        )
    normalized = tuple(normalize_relation(dict(edge)) for edge in record.relations)
    selected = tuple(normalized[index] for index in indices)
    correct = tuple(
        RelationFact(
            subject=alias_by_object_id[str(edge["subject"])],
            predicate=str(edge["relation"]),
            object=alias_by_object_id[str(edge["object"])],
        )
        for edge in selected
    )
    if str(query.get("answer_field")) == "relation":
        query_pair = (
            alias_by_object_id[str(query["subject"])],
            alias_by_object_id[str(query["object"])],
        )
        if any((fact.subject, fact.object) == query_pair for fact in correct):
            raise ValueError(f"{record.scene_id}: relation evidence directly states target pair")
    truth = _truth_closure(normalized, alias_by_object_id)
    corrupted: list[RelationFact] = []
    methods: list[str] = []
    for position, fact in enumerate(correct):
        candidate, method = _false_fact(
            fact,
            truth,
            seed=_bound_seed(seed, record.scene_id, corruption_salt, position),
        )
        corrupted.append(candidate)
        methods.append(method)
    if any(fact in truth for fact in corrupted):
        raise ValueError(f"{record.scene_id}: relation corruption is accidentally true")
    return (
        RelationIntervention(
            condition="correct_relation",
            facts=correct,
            source_edge_indices=indices,
            corruption_methods=(),
        ),
        RelationIntervention(
            condition="corrupted_relation",
            facts=tuple(corrupted),
            source_edge_indices=indices,
            corruption_methods=tuple(methods),
        ),
    )


def relation_by_condition(
    interventions: tuple[RelationIntervention, RelationIntervention], condition: str
) -> RelationIntervention:
    for intervention in interventions:
        if intervention.condition == condition:
            return intervention
    raise KeyError(condition)


def _false_fact(
    fact: RelationFact, truth: set[RelationFact], *, seed: int
) -> tuple[RelationFact, str]:
    candidates: list[tuple[RelationFact, str]] = []
    for predicate in _PREFERRED_FALSE[fact.predicate]:
        candidates.append((RelationFact(fact.subject, predicate, fact.object), "inverse_or_flip"))
    candidates.append(
        (RelationFact(fact.object, fact.predicate, fact.subject), "ordered_entity_swap")
    )
    for predicate in RELATIONS:
        candidates.append(
            (RelationFact(fact.subject, str(predicate), fact.object), "relation_flip")
        )
    valid = [candidate for candidate in candidates if candidate[0] not in truth]
    if not valid:
        raise ValueError(f"no false relation corruption exists for {fact}")
    rng = random.Random(seed)
    preferred = valid[: min(3, len(valid))]
    return preferred[rng.randrange(len(preferred))]


def _truth_closure(
    edges: tuple[dict[str, str], ...], alias_by_object_id: dict[str, str]
) -> set[RelationFact]:
    truth: set[RelationFact] = set()
    for edge in edges:
        subject = alias_by_object_id[str(edge["subject"])]
        target = alias_by_object_id[str(edge["object"])]
        predicate = str(edge["relation"])
        truth.add(RelationFact(subject, predicate, target))
        inverse = _INVERSE.get(predicate)
        if inverse is not None:
            truth.add(RelationFact(target, inverse, subject))
    return truth


def _bound_seed(seed: int, scene_id: str, salt: str, position: int) -> int:
    payload = f"{seed}|{scene_id}|{salt}|{position}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


__all__ = [
    "RelationFact",
    "RelationIntervention",
    "build_relation_interventions",
    "relation_by_condition",
]
