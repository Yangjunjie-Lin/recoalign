from __future__ import annotations

import pytest

from recoalign.synthetic_world.ontology import (
    CATEGORIES,
    COLORS,
    RELATIONS,
    SHAPES,
    ObjectSpec,
    RelationSpec,
)
from recoalign.synthetic_world.scene_graph import SceneGraph


def _object(identifier: str, shape: str, color: str) -> dict[str, str]:
    return ObjectSpec(identifier, "object", shape, color, "medium", "solid").to_dict()


def test_ontology_contains_all_preregistered_factors_and_relations() -> None:
    assert {"circle", "square", "triangle", "cube", "sphere"} <= set(SHAPES)
    assert {"animal", "vehicle", "object"} <= set(CATEGORIES)
    assert {"red", "blue", "green"} <= set(COLORS)
    assert {
        "left",
        "right",
        "above",
        "below",
        "front",
        "behind",
        "near",
        "far",
        "inside",
        "contains",
        "touching",
        "holding",
    } == set(RELATIONS)
    with pytest.raises(ValueError, match="unknown relation"):
        RelationSpec("a", "adjacent", "b")


def test_scene_graph_resolves_licensed_multihop_and_inverse_relations() -> None:
    nodes = (
        _object("a", "circle", "red"),
        _object("b", "square", "blue"),
        _object("c", "triangle", "green"),
        _object("d", "cube", "yellow"),
        _object("e", "sphere", "purple"),
    )
    edges = tuple(
        RelationSpec(first, "left", second).to_dict()
        for first, second in zip("abcd", "bcde", strict=True)
    )
    graph = SceneGraph(nodes, edges)
    assert graph.relation("a", "e", max_hops=4) == "left"
    assert graph.relation("e", "d") == "right"


def test_nontransitive_relation_is_not_invented() -> None:
    nodes = (
        _object("a", "circle", "red"),
        _object("b", "square", "blue"),
        _object("c", "triangle", "green"),
    )
    graph = SceneGraph(
        nodes,
        (
            RelationSpec("a", "touching", "b").to_dict(),
            RelationSpec("b", "touching", "c").to_dict(),
        ),
    )
    assert graph.relation("a", "c") is None
