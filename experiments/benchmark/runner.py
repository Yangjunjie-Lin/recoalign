"""Run the three Phase-1 experiments and collect a compact summary."""

from __future__ import annotations

from pathlib import Path

from experiments.graph_ablation.runner import run as run_ablation
from experiments.graph_vs_text.runner import run as run_graph_vs_text
from experiments.ood_composition.runner import run as run_ood


def run(config_dir: str | Path = "configs") -> dict[str, object]:
    root = Path(config_dir)
    return {
        "graph_vs_text": run_graph_vs_text(root / "graph_vs_text.yaml"),
        "graph_ablation": run_ablation(root / "graph_ablation.yaml"),
        "ood_composition": run_ood(root / "ood_composition.yaml"),
    }

