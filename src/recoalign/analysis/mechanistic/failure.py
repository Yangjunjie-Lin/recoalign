"""Auditable before/after failure-case collection."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def collect_failure_cases(
    original: list[dict[str, Any]],
    recoalign: list[dict[str, Any]],
    *,
    output_dir: str | Path = "reports/failure_cases",
) -> dict[str, Any]:
    left = {str(row["sample_id"]): row for row in original}
    right = {str(row["sample_id"]): row for row in recoalign}
    cases = []
    before_counts: Counter[str] = Counter()
    after_counts: Counter[str] = Counter()
    for sample_id in sorted(set(left) & set(right)):
        baseline = left[sample_id]
        method = right[sample_id]
        baseline_failure = str(baseline.get("failure_type") or "correct")
        method_failure = str(method.get("failure_type") or "correct")
        if baseline_failure == "correct" and method_failure == "correct":
            continue
        if baseline_failure != "correct":
            before_counts[baseline_failure] += 1
        if method_failure != "correct":
            after_counts[method_failure] += 1
        cases.append(
            {
                "sample_id": sample_id,
                "before": {
                    "correct": bool(baseline.get("correct")),
                    "failure_type": baseline_failure,
                    "prediction": baseline.get("prediction"),
                },
                "after": {
                    "correct": bool(method.get("correct")),
                    "failure_type": method_failure,
                    "prediction": method.get("prediction"),
                },
                "transition": f"{baseline_failure}->{method_failure}",
            }
        )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, sort_keys=True, ensure_ascii=False) + "\n")
    summary = {
        "n_pairs": len(set(left) & set(right)),
        "n_cases": len(cases),
        "before": dict(before_counts),
        "after": dict(after_counts),
        "reduced_total_failures": sum(before_counts.values()) > sum(after_counts.values()),
        "oracle_graph_used": False,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


__all__ = ["collect_failure_cases"]
