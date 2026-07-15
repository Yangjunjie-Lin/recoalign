"""Collect finalized experiment records into reviewable tables."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def collect_runs(
    root: str | Path, statuses: Iterable[str] = ("reportable",)
) -> list[dict[str, Any]]:
    """Collect run metadata and metrics from a results tree."""
    accepted = set(statuses)
    records: list[dict[str, Any]] = []
    for run_file in sorted(Path(root).glob("**/run.json")):
        with run_file.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("status") not in accepted:
            continue
        metrics_path = run_file.parent / str(record.get("metrics_file") or "metrics.json")
        if not metrics_path.is_file():
            continue
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        records.append({**record, **metrics, "run_dir": str(run_file.parent)})
    return records


def render_markdown_table(records: list[dict[str, Any]], metric_names: list[str]) -> str:
    """Render a deterministic Markdown table from run summaries."""
    columns = ["run_id", "model", "dataset", "seed", *metric_names]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, separator]
    for record in records:
        values = [_format_value(record.get(column, "")) for column in columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows) + "\n"


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")
