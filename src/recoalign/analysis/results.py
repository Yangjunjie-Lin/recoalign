"""Collect finalized experiment records into reviewable tables."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from recoalign.experiments.records import load_run
from recoalign.schema_validation import validate_payload


def collect_runs(
    root: str | Path, statuses: Iterable[str] = ("reportable",)
) -> list[dict[str, Any]]:
    """Collect schema-valid run metadata and metrics from a results tree."""
    accepted = set(statuses)
    records: list[dict[str, Any]] = []
    for run_file in sorted(Path(root).glob("**/run.json")):
        record = load_run(run_file.parent)
        if record["status"] not in accepted:
            continue
        if record["status"] == "reportable" and not isinstance(record.get("review"), dict):
            raise ValueError(f"reportable run is missing review metadata: {run_file.parent}")
        metrics_path = run_file.parent / str(record.get("metrics_file") or "metrics.json")
        if not metrics_path.is_file():
            raise FileNotFoundError(f"metrics file is missing: {metrics_path}")
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        validate_payload("metrics", metrics)
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
