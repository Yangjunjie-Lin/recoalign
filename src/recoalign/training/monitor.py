"""Machine-readable training monitoring and dependency-free SVG curves."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recoalign.reproducibility import atomic_write_json, utc_now


class TrainingMonitor:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.training_history = _read_jsonl(self.logs_dir / "training.jsonl")
        self.validation_history = _read_jsonl(self.logs_dir / "validation.jsonl")

    def log_training(self, epoch: int, metrics: dict[str, float]) -> None:
        row = {"recorded_at": utc_now(), "epoch": int(epoch), **metrics}
        self.training_history.append(row)
        _append_jsonl(self.logs_dir / "training.jsonl", row)

    def log_validation(self, epoch: int, metrics: dict[str, float]) -> None:
        row = {"recorded_at": utc_now(), "epoch": int(epoch), **metrics}
        self.validation_history.append(row)
        _append_jsonl(self.logs_dir / "validation.jsonl", row)

    def write_summary(self, payload: dict[str, Any]) -> None:
        atomic_write_json(self.run_dir / "metrics.json", payload)
        self.render_curves()

    def render_curves(self) -> Path:
        path = self.logs_dir / "loss_curves.svg"
        series = {
            name: [float(row[name]) for row in self.training_history if name in row]
            for name in (
                "total_loss",
                "semantic_loss",
                "structural_loss",
                "reasoning_loss",
            )
        }
        path.write_text(_loss_svg(series), encoding="utf-8")
        return path


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _loss_svg(series: dict[str, list[float]]) -> str:
    width, height, pad = 720, 420, 52
    values = [value for rows in series.values() for value in rows]
    maximum = max(values, default=1.0)
    minimum = min(values, default=0.0)
    span = maximum - minimum or 1.0
    longest = max((len(rows) for rows in series.values()), default=1)
    colors = {
        "total_loss": "#111827",
        "semantic_loss": "#2563eb",
        "structural_loss": "#dc2626",
        "reasoning_loss": "#059669",
    }
    polylines: list[str] = []
    legends: list[str] = []
    for index, (name, rows) in enumerate(series.items()):
        points = []
        for step, value in enumerate(rows):
            x = pad + (width - 2 * pad) * step / max(1, longest - 1)
            y = height - pad - (height - 2 * pad) * (value - minimum) / span
            points.append(f"{x:.1f},{y:.1f}")
        if points:
            polylines.append(
                f'<polyline fill="none" stroke="{colors[name]}" stroke-width="2" '
                f'points="{" ".join(points)}" />'
            )
        legends.append(
            f'<text x="{pad + index * 150}" y="24" fill="{colors[name]}" '
            f'font-size="12">{name}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n'
        '<rect width="100%" height="100%" fill="white" />\n'
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" '
        'stroke="#9ca3af" />\n'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" '
        'stroke="#9ca3af" />\n'
        + "\n".join(legends + polylines)
        + "\n</svg>\n"
    )


__all__ = ["TrainingMonitor"]
