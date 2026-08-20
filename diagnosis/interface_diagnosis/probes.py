"""Dependency-light probes for the three-stage interface diagnosis."""

from __future__ import annotations

import json
from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np


def linear_probe(
    features: np.ndarray,
    labels: Sequence[Hashable],
    *,
    seed: int = 7,
    test_fraction: float = 0.3,
    ridge: float = 1e-3,
) -> dict[str, Any]:
    """Fit a deterministic one-vs-rest linear probe on a scene-disjoint split."""

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(labels) or len(values) < 4:
        raise ValueError("linear probe requires a 2D feature matrix with at least four rows")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one")
    encoded = [str(label) for label in labels]
    classes = sorted(set(encoded))
    if len(classes) < 2:
        return {"status": "unavailable", "reason": "probe requires at least two classes"}
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(values))
    test_size = max(1, min(len(values) - 1, int(round(len(values) * test_fraction))))
    test_idx = order[:test_size]
    train_idx = order[test_size:]
    if len(train_idx) < 2 or len({encoded[index] for index in train_idx}) < 2:
        return {"status": "unavailable", "reason": "train split does not contain two classes"}
    train = np.concatenate((values[train_idx], np.ones((len(train_idx), 1))), axis=1)
    test = np.concatenate((values[test_idx], np.ones((len(test_idx), 1))), axis=1)
    targets = np.zeros((len(train_idx), len(classes)), dtype=np.float64)
    class_index = {label: index for index, label in enumerate(classes)}
    for row, index in enumerate(train_idx):
        targets[row, class_index[encoded[index]]] = 1.0
    gram = train.T @ train + ridge * np.eye(train.shape[1])
    weights = np.linalg.solve(gram, train.T @ targets)
    scores = test @ weights
    predictions = [classes[index] for index in np.argmax(scores, axis=1)]
    truth = [encoded[index] for index in test_idx]
    accuracy = sum(
        prediction == actual for prediction, actual in zip(predictions, truth, strict=True)
    ) / len(truth)
    return {
        "status": "available",
        "accuracy": float(accuracy),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_classes": len(classes),
        "classes": classes,
        "seed": seed,
    }


def graph_score(predicted: Any, truth: dict[str, Any]) -> dict[str, Any]:
    """Score node/relation F1 and graph edit distance from JSON-like graph payloads."""

    expected_nodes = {_node_key(node) for node in truth.get("objects", truth.get("nodes", []))}
    expected_edges = {_edge_key(edge) for edge in truth.get("relations", truth.get("edges", []))}
    actual = _coerce_graph(predicted)
    actual_nodes = {_node_key(node) for node in actual.get("objects", actual.get("nodes", []))}
    actual_edges = {_edge_key(edge) for edge in actual.get("relations", actual.get("edges", []))}
    node_f1 = _f1(actual_nodes, expected_nodes)
    relation_f1 = _f1(actual_edges, expected_edges)
    edit_distance = float(len(actual_nodes ^ expected_nodes) + len(actual_edges ^ expected_edges))
    return {
        "status": "available",
        "node_f1": node_f1,
        "relation_f1": relation_f1,
        "graph_edit_distance": edit_distance,
        "exact": actual_nodes == expected_nodes and actual_edges == expected_edges,
    }


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float).reshape(-1)
    right = np.asarray(second, dtype=float).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def consistency_score(features: np.ndarray, labels: Sequence[Hashable]) -> dict[str, Any]:
    """Compare same-relation and different-relation cosine similarity."""

    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or len(values) != len(labels) or len(values) < 4:
        return {"status": "unavailable", "reason": "at least four aligned representations required"}
    same: list[float] = []
    different: list[float] = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            similarity = cosine_similarity(values[left], values[right])
            (same if labels[left] == labels[right] else different).append(similarity)
    if not same or not different:
        return {"status": "unavailable", "reason": "both matched and mismatched pairs required"}
    gap = float(np.mean(same) - np.mean(different))
    return {
        "status": "available",
        "same_relation_similarity": float(np.mean(same)),
        "different_relation_similarity": float(np.mean(different)),
        "consistency_gap": gap,
        "n_same_pairs": len(same),
        "n_different_pairs": len(different),
    }


def _coerce_graph(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        if isinstance(value.get("scene_graph"), dict):
            return value["scene_graph"]
        if isinstance(value.get("graph"), dict):
            value = value["graph"]
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"objects": [], "relations": []}
        return _coerce_graph(parsed)
    return {"objects": [], "relations": []}


def _node_key(node: Any) -> str:
    if not isinstance(node, dict):
        return str(node)
    return "|".join(
        str(node.get(key, ""))
        for key in ("id", "object_id", "category", "shape", "color", "size", "texture")
    )


def _edge_key(edge: Any) -> str:
    if not isinstance(edge, dict):
        return str(edge)
    return "|".join(
        str(edge.get(key, edge.get(alias, "")))
        for key, alias in (("subject", "source"), ("relation", "predicate"), ("object", "target"))
    )


def _f1(predicted: set[str], truth: set[str]) -> float:
    if not predicted and not truth:
        return 1.0
    if not predicted or not truth:
        return 0.0
    precision = len(predicted & truth) / len(predicted)
    recall = len(predicted & truth) / len(truth)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


__all__ = ["consistency_score", "cosine_similarity", "graph_score", "linear_probe"]
