"""Slice condition predictions by seed, relation, and composition split."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def accuracy_by(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field, "unknown"))].append(bool(row["correct"]))
    return {key: sum(values) / len(values) for key, values in sorted(grouped.items()) if values}
