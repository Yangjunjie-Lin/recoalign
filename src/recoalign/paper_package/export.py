# ruff: noqa: E501
"""Honest paper table and figure export from retained evidence."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from .common import load_json, project_root, write_text

PENDING = "Evidence pending under frozen protocol; no paper value is reported."


def export_results(root: str | Path | None = None) -> dict[str, Any]:
    project = project_root(root)
    results = project / "reports/results"
    latex = project / "reports/latex"
    figures = project / "reports/figures"
    for directory in (results, latex, figures):
        directory.mkdir(parents=True, exist_ok=True)

    decision_path = project / "reports/decision_report.json"
    decision = load_json(decision_path) if decision_path.is_file() else {}
    complete = int(decision.get("complete_cells", 0))
    planned = int(decision.get("planned_cells", 432))

    tables = {
        "main_results.tex": _pending_table(
            "Main multi-VLM results",
            "Model & Benchmark & Baseline & ReCoAlign",
            f"Frozen matrix completion: {complete}/{planned}. {PENDING}",
        ),
        "ablation_results.tex": _pending_table(
            "Mechanistic ablations",
            "Ablation & Variable & Seeds & Effect",
            f"Real-VLM ablation matrix incomplete. {PENDING}",
        ),
        "ood_results.tex": _pending_table(
            "OOD compositional generalization",
            "Model & IID & OOD & Retention",
            f"Claim-eligible EXP003 results incomplete. {PENDING}",
        ),
        "analysis_results.tex": _pending_table(
            "Interface diagnosis and causal analysis",
            "Model & SAS & StAS & RES",
            f"Real hidden-state diagnosis incomplete. {PENDING}",
        ),
    }
    for name, content in tables.items():
        write_text(results / name, content)
        write_text(latex / name, content)

    _write_figure_overview(figures / "figure_1_research_overview.svg")
    _write_figure_mechanism(figures / "figure_2_mechanism_validation.svg")
    _write_figure_architecture(figures / "figure_3_method_architecture.svg")
    _write_figure_benchmarks(figures / "figure_4_benchmark_results.svg", complete, planned)
    _write_figure_analysis(figures / "figure_5_mechanistic_analysis.svg")

    manifest = {
        "tables": [f"reports/results/{name}" for name in tables],
        "latex_mirrors": [f"reports/latex/{name}" for name in tables],
        "figures": [
            f"reports/figures/figure_{index}_{stem}.svg"
            for index, stem in (
                (1, "research_overview"),
                (2, "mechanism_validation"),
                (3, "method_architecture"),
                (4, "benchmark_results"),
                (5, "mechanistic_analysis"),
            )
        ],
        "complete_matrix_cells": complete,
        "planned_matrix_cells": planned,
        "contains_placeholder_numbers": False,
        "pending_cells_explicit": True,
    }
    return manifest


def _pending_table(caption: str, headers: str, message: str) -> str:
    columns = headers.count("&") + 1
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\begin{{tabular}}{{{'l' * columns}}}",
            r"\toprule",
            headers + r" \\",
            r"\midrule",
            rf"\multicolumn{{{columns}}}{{l}}{{\textit{{{message}}}}} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def _svg(
    path: Path,
    title: str,
    subtitle: str,
    boxes: list[tuple[int, int, int, int, str, str]],
    arrows: list[tuple[int, int, int, int]],
) -> None:
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="520" viewBox="0 0 1200 520">',
        '<rect width="1200" height="520" fill="#fbfcfe"/>',
        f'<text x="60" y="58" font-family="Arial" font-size="28" font-weight="700" fill="#172033">{html.escape(title)}</text>',
        f'<text x="60" y="88" font-family="Arial" font-size="15" fill="#596579">{html.escape(subtitle)}</text>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#53637a"/></marker></defs>',
    ]
    for x1, y1, x2, y2 in arrows:
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#53637a" stroke-width="3" marker-end="url(#arrow)"/>'
        )
    for x, y, width, height, label, color in boxes:
        parts.extend(
            [
                f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="14" fill="{color}" stroke="#26364d" stroke-width="2"/>',
                f'<text x="{x + width / 2}" y="{y + height / 2 + 6}" text-anchor="middle" font-family="Arial" font-size="18" font-weight="600" fill="#172033">{html.escape(label)}</text>',
            ]
        )
    parts.append("</svg>")
    write_text(path, "\n".join(parts))


def _write_figure_overview(path: Path) -> None:
    boxes = [
        (60, 180, 210, 90, "Visual semantics", "#d9edff"),
        (360, 180, 210, 90, "Interface gap", "#ffe2dc"),
        (660, 180, 210, 90, "ReCoAlign", "#dcf7e8"),
        (960, 180, 180, 90, "Reasoning", "#eee5ff"),
    ]
    _svg(
        path,
        "ReCoAlign research overview",
        "Diagnosis-driven method development",
        boxes,
        [(270, 225, 350, 225), (570, 225, 650, 225), (870, 225, 950, 225)],
    )


def _write_figure_mechanism(path: Path) -> None:
    boxes = [
        (70, 160, 220, 90, "EXP001 Graph > Text", "#d9edff"),
        (360, 160, 220, 90, "EXP002 Corruption", "#ffe9c8"),
        (650, 160, 220, 90, "EXP003 OOD", "#dcf7e8"),
        (940, 160, 200, 90, "Real VLM pending", "#ffe2dc"),
    ]
    _svg(
        path,
        "Mechanism validation chain",
        "Protocols implemented; claim-eligible real-VLM results pending",
        boxes,
        [(290, 205, 350, 205), (580, 205, 640, 205), (870, 205, 930, 205)],
    )


def _write_figure_architecture(path: Path) -> None:
    boxes = [
        (70, 180, 200, 90, "Vision encoder", "#d9edff"),
        (360, 180, 220, 90, "Visual tokens", "#e9f1fb"),
        (670, 160, 230, 130, "Structured interface", "#dcf7e8"),
        (990, 180, 150, 90, "LLM", "#eee5ff"),
    ]
    _svg(
        path,
        "ReCoAlign architecture",
        "Oracle graphs are training signals only and never inference inputs",
        boxes,
        [(270, 225, 350, 225), (580, 225, 660, 225), (900, 225, 980, 225)],
    )


def _write_figure_benchmarks(path: Path, complete: int, planned: int) -> None:
    boxes = [
        (90, 170, 260, 100, "4 VLM backbones", "#d9edff"),
        (470, 170, 260, 100, "9 benchmarks", "#dcf7e8"),
        (850, 150, 260, 140, f"Complete: {complete}/{planned}", "#ffe2dc"),
    ]
    _svg(
        path,
        "Frozen benchmark matrix",
        "Missing cells remain visible; no synthetic result values are inserted",
        boxes,
        [(350, 220, 460, 220), (730, 220, 840, 220)],
    )


def _write_figure_analysis(path: Path) -> None:
    boxes = [
        (70, 160, 210, 100, "Representation probes", "#d9edff"),
        (360, 160, 210, 100, "Token interventions", "#ffe9c8"),
        (650, 160, 210, 100, "Parameter control", "#dcf7e8"),
        (940, 160, 200, 100, "Real evidence pending", "#ffe2dc"),
    ]
    _svg(
        path,
        "Mechanistic analysis",
        "Toy controls validate implementation, not the paper claim",
        boxes,
        [(280, 210, 350, 210), (570, 210, 640, 210), (860, 210, 930, 210)],
    )
