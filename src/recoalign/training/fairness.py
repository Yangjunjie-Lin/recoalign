"""Fair baseline-matrix validation for training comparisons."""

from __future__ import annotations

from typing import Any

REQUIRED_BASELINES = (
    "original_vlm",
    "caption_adapter",
    "oracle_graph_prompt",
    "recoalign",
)
FAIRNESS_FIELDS = ("vision_backbone", "llm_backbone", "dataset", "split", "seed")


def validate_baseline_fairness(matrix: dict[str, dict[str, Any]]) -> dict[str, Any]:
    missing = sorted(set(REQUIRED_BASELINES) - set(matrix))
    if missing:
        raise ValueError(f"baseline matrix is missing: {', '.join(missing)}")
    reference = matrix["recoalign"]
    mismatches: list[dict[str, Any]] = []
    for name in REQUIRED_BASELINES:
        row = matrix[name]
        for field in FAIRNESS_FIELDS:
            if row.get(field) != reference.get(field):
                mismatches.append(
                    {
                        "baseline": name,
                        "field": field,
                        "expected": reference.get(field),
                        "actual": row.get(field),
                    }
                )
    if mismatches:
        rendered = "; ".join(
            f"{row['baseline']}.{row['field']}={row['actual']!r} "
            f"(expected {row['expected']!r})"
            for row in mismatches
        )
        raise ValueError(f"unfair baseline matrix: {rendered}")
    if matrix["recoalign"].get("oracle_graph_at_inference") is not False:
        raise ValueError("ReCoAlign must set oracle_graph_at_inference=false")
    if matrix["oracle_graph_prompt"].get("oracle_graph_at_inference") is not True:
        raise ValueError("oracle graph baseline must be explicitly marked")
    return {
        "valid": True,
        "baselines": list(REQUIRED_BASELINES),
        "controlled_fields": list(FAIRNESS_FIELDS),
        "recoalign_oracle_graph_at_inference": False,
    }


__all__ = ["FAIRNESS_FIELDS", "REQUIRED_BASELINES", "validate_baseline_fairness"]
