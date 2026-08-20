"""Deterministic IID and factor-held-out split policies."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

from datasets.records import SceneRecord

SPLIT_STRATEGIES = ("iid", "composition", "relation", "attribute")


def composition_signature(record: SceneRecord) -> str:
    query = dict(record.metadata.get("query", {}))
    lookup = {str(node["id"]): node for node in record.objects}
    subject = lookup[str(query.get("subject", record.objects[0]["id"]))]
    target = lookup[str(query.get("object", record.objects[-1]["id"]))]
    relation = str(record.metadata.get("primary_relation", record.relations[0]["relation"]))
    return "|".join(
        (
            _factor_tuple(subject),
            relation,
            _factor_tuple(target),
            str(record.metadata.get("question_type", "unknown")),
        )
    )


def assign_split(record: SceneRecord, strategy: str) -> str:
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"split strategy must be one of {SPLIT_STRATEGIES}")
    if strategy == "iid":
        bucket = _bucket(record.scene_id, 100)
        return "train" if bucket < 80 else "validation" if bucket < 90 else "test"
    query = dict(record.metadata.get("query", {}))
    lookup = {str(node["id"]): node for node in record.objects}
    subject = lookup[str(query.get("subject", record.objects[0]["id"]))]
    target = lookup[str(query.get("object", record.objects[-1]["id"]))]
    if strategy == "relation":
        relation = str(record.metadata.get("primary_relation", record.relations[0]["relation"]))
        if relation in {"left", "right"}:
            return "train"
        if relation in {"front", "behind"}:
            return "test"
        return "validation"
    if strategy == "attribute":
        color = str(subject["color"])
        if color in {"red", "blue"}:
            return "train"
        if color == "green":
            return "test"
        return "validation"
    first = _factor_tuple(subject)
    second = _factor_tuple(target)
    family = "|".join(
        (
            min(first, second),
            str(record.metadata.get("primary_relation", record.relations[0]["relation"])),
            max(first, second),
            str(record.metadata.get("question_type", "unknown")),
        )
    )
    if _bucket(family, 10) == 0:
        return "validation"
    return "train" if first < second else "test"


def apply_split(records: list[SceneRecord], strategy: str) -> list[SceneRecord]:
    assigned: list[SceneRecord] = []
    for record in records:
        split = assign_split(record, strategy)
        metadata: dict[str, Any] = {
            **record.metadata,
            "split": split,
            "split_strategy": strategy,
            "composition": composition_signature(record),
        }
        assigned.append(replace(record, split=split, metadata=metadata))
    return assigned


def split_leakage_report(records: list[SceneRecord], strategy: str) -> dict[str, Any]:
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"split strategy must be one of {SPLIT_STRATEGIES}")
    by_split = {
        name: [record for record in records if record.split == name]
        for name in ("train", "validation", "test")
    }
    report: dict[str, Any] = {
        "strategy": strategy,
        "counts": {name: len(rows) for name, rows in by_split.items()},
        "valid": True,
        "violations": [],
    }
    misassigned = sorted(
        record.scene_id
        for record in records
        if assign_split(record, strategy) != record.split
    )
    report["misassigned_samples"] = misassigned
    if misassigned:
        report["violations"].append("records do not match deterministic split assignment")
    if strategy == "composition":
        train = {composition_signature(row) for row in by_split["train"]}
        test = {composition_signature(row) for row in by_split["test"]}
        overlap = sorted(train & test)
        report["composition_overlap"] = overlap
        if overlap:
            report["violations"].append("train/test composition overlap")
    elif strategy == "relation":
        train_relations = {_primary_relation(row) for row in by_split["train"]}
        test_relations = {_primary_relation(row) for row in by_split["test"]}
        report["train_relations"] = sorted(train_relations)
        report["test_relations"] = sorted(test_relations)
        if not train_relations <= {"left", "right"} or not test_relations <= {"front", "behind"}:
            report["violations"].append("relation holdout policy violated")
    elif strategy == "attribute":
        train_colors = {_query_subject(row)["color"] for row in by_split["train"]}
        test_colors = {_query_subject(row)["color"] for row in by_split["test"]}
        report["train_colors"] = sorted(train_colors)
        report["test_colors"] = sorted(test_colors)
        if not train_colors <= {"red", "blue"} or not test_colors <= {"green"}:
            report["violations"].append("attribute holdout policy violated")
    else:
        ids = [row.scene_id for rows in by_split.values() for row in rows]
        if len(ids) != len(set(ids)):
            report["violations"].append("sample assigned to multiple IID partitions")
    report["valid"] = not report["violations"]
    return report


def _query_subject(record: SceneRecord) -> dict[str, Any]:
    query = dict(record.metadata.get("query", {}))
    identifier = str(query.get("subject", record.objects[0]["id"]))
    return next(node for node in record.objects if str(node["id"]) == identifier)


def _primary_relation(record: SceneRecord) -> str:
    return str(record.metadata.get("primary_relation", record.relations[0]["relation"]))


def _factor_tuple(node: dict[str, Any]) -> str:
    return ":".join(
        str(node[field]) for field in ("color", "shape", "size", "texture", "category")
    )


def _bucket(value: str, modulus: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


__all__ = [
    "SPLIT_STRATEGIES",
    "apply_split",
    "assign_split",
    "composition_signature",
    "split_leakage_report",
]
