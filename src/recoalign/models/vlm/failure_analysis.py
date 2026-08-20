"""Automatic error taxonomy for real-VLM structured-reasoning runs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def write_failure_analysis(
    predictions_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    source = Path(predictions_path)
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_scene: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_scene[str(row.get("scene_id", row.get("sample_id")))][str(row["condition"])] = row
    failures: list[dict[str, Any]] = []
    for row in rows:
        if bool(row.get("correct")):
            continue
        scene_id = str(row.get("scene_id", row.get("sample_id")))
        failure_type = _failure_type(row, by_scene[scene_id])
        failures.append(
            {
                "sample_id": scene_id,
                "condition": row.get("condition"),
                "failure_type": failure_type,
                "question_type": row.get("question_type"),
                "hop_depth": row.get("hop_depth"),
                "prediction": row.get("prediction"),
                "ground_truth": row.get("ground_truth", row.get("answer")),
                "split": row.get("split"),
            }
        )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "errors.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in failures:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    counts = Counter(str(row["failure_type"]) for row in failures)
    by_condition = Counter(str(row["condition"]) for row in failures)
    summary = {
        "schema_version": 1,
        "prediction_count": len(rows),
        "failure_count": len(failures),
        "failure_rate": len(failures) / len(rows) if rows else 0.0,
        "failure_types": dict(sorted(counts.items())),
        "conditions": dict(sorted(by_condition.items())),
        "artifacts": {"errors": "errors.jsonl", "summary": "summary.json"},
    }
    _write_json(destination / "summary.json", summary)
    lines = [
        "# VLM Failure Analysis",
        "",
        f"- Predictions: {len(rows)}",
        f"- Failures: {len(failures)}",
        f"- Failure rate: {summary['failure_rate']:.4f}",
        "",
        "| Failure type | Count |",
        "|---|---:|",
        *[f"| {name} | {count} |" for name, count in sorted(counts.items())],
        "",
    ]
    (destination / "report.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )
    return summary


def _failure_type(
    row: dict[str, Any], scene_conditions: dict[str, dict[str, Any]]
) -> str:
    condition = str(row.get("condition"))
    if condition == "scene_graph":
        caption = scene_conditions.get("caption")
        if caption is not None and bool(caption.get("correct")):
            return "structure_misuse"
    question_type = str(row.get("question_type", ""))
    if question_type in {"object_reasoning", "attribute_reasoning"}:
        return "object_failure"
    if question_type == "multi_hop" or int(row.get("hop_depth") or 0) > 1:
        return "multi_hop_failure"
    return "relation_failure"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


__all__ = ["write_failure_analysis"]
