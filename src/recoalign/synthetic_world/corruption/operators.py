"""Deterministic, auditable interventions on oracle scene graphs for EXP002."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any

from recoalign.synthetic_world.ontology import RELATIONS, normalize_relation
from recoalign.synthetic_world.scene_graph import SceneGraph

SEMANTIC_FLIPS = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
    "front": "behind",
    "behind": "front",
    "near": "far",
    "far": "near",
    "inside": "contains",
    "contains": "inside",
    "touching": "holding",
    "holding": "touching",
}


@dataclass(frozen=True)
class CorruptionResult:
    """A graph intervention and the complete provenance needed to reconstruct it."""

    original: SceneGraph
    corrupted: SceneGraph
    operation: str
    seed: int
    ratio: float
    selected_edge_indices: tuple[int, ...]
    details: dict[str, Any] = field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        original = self.original.to_dict()
        corrupted = self.corrupted.to_dict()
        return {
            "original_graph": original,
            "corrupted_graph": corrupted,
            "operation": self.operation,
            "seed": self.seed,
            "ratio": self.ratio,
            "selected_edge_indices": list(self.selected_edge_indices),
            "original_graph_sha256": _graph_sha256(original),
            "corrupted_graph_sha256": _graph_sha256(corrupted),
            "preserves_nodes": original["objects"] == corrupted["objects"],
            "preserves_edge_count": len(original["relations"])
            == len(corrupted["relations"]),
            "changed": original != corrupted,
            "details": dict(self.details),
        }


def remove_relation(
    graph: SceneGraph,
    *,
    ratio: float,
    seed: int = 0,
    critical_edge_indices: tuple[int, ...] | list[int] = (),
) -> CorruptionResult:
    """Remove a controlled fraction of edges, prioritizing registered supporting edges."""

    _validate_ratio(ratio)
    if not graph.edges:
        raise ValueError("remove_relation requires at least one edge")
    count = min(len(graph.edges), max(1, math.ceil(len(graph.edges) * ratio)))
    rng = random.Random(seed)
    critical = sorted({int(index) for index in critical_edge_indices})
    if any(index < 0 or index >= len(graph.edges) for index in critical):
        raise ValueError("critical edge index is outside the graph")
    remainder = [index for index in range(len(graph.edges)) if index not in critical]
    rng.shuffle(critical)
    rng.shuffle(remainder)
    selected = tuple(sorted((*critical, *remainder)[:count]))
    kept = tuple(edge for index, edge in enumerate(graph.edges) if index not in selected)
    return CorruptionResult(
        graph,
        SceneGraph(graph.nodes, kept),
        "remove_relation",
        seed,
        ratio,
        selected,
        {
            "requested_ratio": ratio,
            "realized_ratio": len(selected) / len(graph.edges),
            "critical_edges_prioritized": bool(critical),
        },
    )


def flip_relation(
    graph: SceneGraph, *, ratio: float = 1.0, seed: int = 0
) -> CorruptionResult:
    """Replace selected predicates with their declared semantic opposites."""

    selected = _selected_indices(graph, ratio, seed)
    edges = [normalize_relation(dict(edge)) for edge in graph.edges]
    replacements: list[dict[str, str]] = []
    for index in selected:
        before = edges[index]["relation"]
        after = SEMANTIC_FLIPS[before]
        edges[index]["relation"] = after
        replacements.append({"before": before, "after": after})
    return CorruptionResult(
        graph,
        SceneGraph(graph.nodes, tuple(edges)),
        "relation_flip",
        seed,
        ratio,
        selected,
        {"replacements": replacements},
    )


def swap_entity(
    graph: SceneGraph, *, ratio: float = 1.0, seed: int = 0
) -> CorruptionResult:
    """Change one endpoint per selected edge while keeping the node inventory fixed."""

    selected = _selected_indices(graph, ratio, seed)
    rng = random.Random(seed + 17)
    identifiers = [str(node["id"]) for node in graph.nodes]
    edges = [normalize_relation(dict(edge)) for edge in graph.edges]
    replacements: list[dict[str, str]] = []
    for index in selected:
        edge = edges[index]
        field = ("subject", "object")[rng.randrange(2)]
        other = "object" if field == "subject" else "subject"
        candidates = [
            value for value in identifiers if value not in {edge[field], edge[other]}
        ]
        before = edge[field]
        if candidates:
            edge[field] = candidates[rng.randrange(len(candidates))]
            after = edge[field]
        else:
            edge["subject"], edge["object"] = edge["object"], edge["subject"]
            field = "both"
            after = f"{edge['subject']}|{edge['object']}"
        replacements.append({"field": field, "before": before, "after": after})
    return CorruptionResult(
        graph,
        SceneGraph(graph.nodes, tuple(edges)),
        "entity_swap",
        seed,
        ratio,
        selected,
        {"replacements": replacements},
    )


def randomize_graph(graph: SceneGraph, *, seed: int = 0) -> CorruptionResult:
    """Generate a same-size random relation graph over exactly the original nodes."""

    if not graph.edges:
        raise ValueError("randomize_graph requires at least one edge")
    rng = random.Random(seed)
    identifiers = [str(node["id"]) for node in graph.nodes]
    original = [normalize_relation(dict(edge)) for edge in graph.edges]
    randomized: list[dict[str, str]] = []
    for edge in original:
        candidates = [
            (subject, relation, object_id)
            for subject in identifiers
            for object_id in identifiers
            if subject != object_id
            for relation in RELATIONS
            if (subject, relation, object_id)
            != (edge["subject"], edge["relation"], edge["object"])
        ]
        subject, relation, object_id = candidates[rng.randrange(len(candidates))]
        randomized.append(
            {"subject": subject, "relation": relation, "object": object_id}
        )
    corrupted = SceneGraph(graph.nodes, tuple(randomized))
    return CorruptionResult(
        graph,
        corrupted,
        "random_graph",
        seed,
        1.0,
        tuple(range(len(graph.edges))),
        {"sampling": "uniform_over_valid_nonidentical_triples"},
    )


def _selected_indices(graph: SceneGraph, ratio: float, seed: int) -> tuple[int, ...]:
    _validate_ratio(ratio)
    if not graph.edges:
        raise ValueError("graph corruption requires at least one edge")
    count = min(len(graph.edges), max(1, math.ceil(len(graph.edges) * ratio)))
    indices = list(range(len(graph.edges)))
    random.Random(seed).shuffle(indices)
    return tuple(sorted(indices[:count]))


def _validate_ratio(ratio: float) -> None:
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise ValueError("corruption ratio must be numeric")
    if not 0.0 < float(ratio) <= 1.0:
        raise ValueError("corruption ratio must be in (0, 1]")


def _graph_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "CorruptionResult",
    "SEMANTIC_FLIPS",
    "flip_relation",
    "randomize_graph",
    "remove_relation",
    "swap_entity",
]
