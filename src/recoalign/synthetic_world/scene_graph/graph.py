"""Oracle scene graphs and explicitly licensed symbolic inference rules."""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from recoalign.synthetic_world.ontology import (
    INVERSE_RELATIONS,
    RELATIONS,
    TRANSITIVE_RELATIONS,
    normalize_relation,
    validate_object,
)


@dataclass(frozen=True)
class SceneGraph:
    """Ground-truth nodes and typed edges, never a model prediction."""

    nodes: tuple[dict[str, Any], ...]
    edges: tuple[dict[str, str], ...]

    def __post_init__(self) -> None:
        identifiers: set[str] = set()
        for node in self.nodes:
            validate_object(node)
            identifier = str(node["id"])
            if identifier in identifiers:
                raise ValueError(f"duplicate object ID: {identifier}")
            identifiers.add(identifier)
        for raw in self.edges:
            edge = normalize_relation(raw)
            if edge["subject"] not in identifiers or edge["object"] not in identifiers:
                raise ValueError(f"relation references an unknown object: {edge}")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SceneGraph:
        nodes = payload.get("objects", payload.get("nodes", []))
        edges = payload.get("relations", payload.get("edges", []))
        return cls(
            tuple(dict(node) for node in nodes),
            tuple(normalize_relation(dict(edge)) for edge in edges),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "objects": [dict(node) for node in self.nodes],
            "relations": [normalize_relation(edge) for edge in self.edges],
        }

    def legacy_dict(self) -> dict[str, Any]:
        return {
            "nodes": [dict(node) for node in self.nodes],
            "edges": [normalize_relation(edge) for edge in self.edges],
        }

    def relation(self, source: str, target: str, *, max_hops: int = 4) -> str | None:
        """Resolve direct, inverse, or same-relation transitive paths only."""

        canonical = tuple(normalize_relation(edge) for edge in self.edges)
        for edge in canonical:
            if edge["subject"] == source and edge["object"] == target:
                return edge["relation"]
            if edge["subject"] == target and edge["object"] == source:
                inverse = INVERSE_RELATIONS.get(edge["relation"])
                if inverse is not None:
                    return inverse
        for relation in RELATIONS:
            if relation not in TRANSITIVE_RELATIONS:
                continue
            adjacency: dict[str, list[str]] = {}
            for edge in canonical:
                if edge["relation"] == relation:
                    adjacency.setdefault(edge["subject"], []).append(edge["object"])
                inverse = INVERSE_RELATIONS.get(edge["relation"])
                if inverse == relation:
                    adjacency.setdefault(edge["object"], []).append(edge["subject"])
            queue: deque[tuple[str, int]] = deque([(source, 0)])
            seen = {source}
            while queue:
                current, depth = queue.popleft()
                if depth >= max_hops:
                    continue
                for neighbor in adjacency.get(current, []):
                    if neighbor == target:
                        return relation
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append((neighbor, depth + 1))
        return None

    def text(self, *, edge_indices: Iterable[int] | None = None) -> str:
        indices = list(range(len(self.edges))) if edge_indices is None else list(edge_indices)
        nodes = "\n".join(
            f"- {node['id']}: {node['size']} {node['texture']} {node['color']} "
            f"{node['shape']} [{node['category']}]"
            for node in self.nodes
        )
        edges = "\n".join(
            f"- {edge['subject']} --{edge['relation']}--> {edge['object']}"
            for index in indices
            if 0 <= index < len(self.edges)
            for edge in (normalize_relation(self.edges[index]),)
        )
        return f"Nodes:\n{nodes}\nEdges (Relations):\n{edges}"

    def partial(self) -> SceneGraph:
        keep = max(0, len(self.edges) - 1)
        return SceneGraph(self.nodes, self.edges[:keep])

    def corrupted(self, seed: int = 0) -> SceneGraph:
        if not self.edges:
            return self
        rng = random.Random(seed)
        edge_index = rng.randrange(len(self.edges))
        edge = normalize_relation(self.edges[edge_index])
        alternatives = [value for value in RELATIONS if value != edge["relation"]]
        edge["relation"] = alternatives[rng.randrange(len(alternatives))]
        edges = [normalize_relation(item) for item in self.edges]
        edges[edge_index] = edge
        return SceneGraph(self.nodes, tuple(edges))


__all__ = ["RELATIONS", "SceneGraph"]
