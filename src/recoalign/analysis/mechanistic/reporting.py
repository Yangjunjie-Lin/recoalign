"""Automated ablation/mechanism/visualization/LaTeX report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from recoalign.reproducibility import atomic_write_json, utc_now


def generate_mechanistic_reports(
    suite: dict[str, Any] | str | Path,
    *,
    output_root: str | Path = "reports",
) -> dict[str, Any]:
    payload = _load(suite)
    destination = Path(output_root)
    ablation = destination / "ablation"
    mechanism = destination / "mechanism"
    visualization = destination / "visualization"
    latex = destination / "latex_tables"
    for path in (ablation, mechanism, visualization, latex):
        path.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, values in payload.get("ablation_statistics", {}).items():
        rows.append(
            (
                name,
                values["full"]["mean"],
                values["ablation"]["mean"],
                values["full_minus_ablation"]["mean"],
                values["n_seeds"],
            )
        )
    _write_markdown(
        ablation / "ablation_results.md",
        ("Ablation", "Full mean", "Ablation mean", "Delta", "Seeds"),
        rows,
    )
    _write_latex(
        latex / "table_mechanistic_ablation.tex",
        "Mechanistic ablation results",
        ("Ablation", "Full", "Ablation", "Delta", "Seeds"),
        rows,
    )
    intervention_rows = []
    for row in payload.get("causal_interventions", []):
        for name, result in row.get("interventions", {}).items():
            intervention_rows.append((row.get("seed"), name, result.get("accuracy_delta")))
    _write_markdown(
        mechanism / "causal_interventions.md",
        ("Seed", "Intervention", "Accuracy delta"),
        intervention_rows,
    )
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "status": payload.get("scientific_status", "unknown"),
        "ablation_rows": len(rows),
        "intervention_rows": len(intervention_rows),
        "oracle_graph_used": payload.get("oracle_graph_used", True),
        "scientific_claim_eligible": False,
    }
    atomic_write_json(mechanism / "mechanistic_report.json", report)
    decision = _decision(payload)
    (mechanism / "decision_report.yaml").write_text(
        yaml.safe_dump(decision, sort_keys=False), encoding="utf-8"
    )
    _write_svg(visualization / "figure_structure_token_interventions.svg")
    return report


def _decision(payload: dict[str, Any]) -> dict[str, Any]:
    interventions = payload.get("causal_interventions", [])
    removal_deltas = [
        float(row["interventions"]["remove"]["accuracy_delta"])
        for row in interventions
        if "remove" in row.get("interventions", {})
    ]
    random_delta = payload.get("ablation_statistics", {}).get("random_structure", {}).get(
        "full_minus_ablation", {}
    ).get("mean")
    criteria = {
        "structure_removal_hurts": (
            bool(removal_deltas) and sum(removal_deltas) / len(removal_deltas) < 0
        ),
        "random_tokens_fail": isinstance(random_delta, (int, float)) and random_delta > 0,
        "representation_evidence": bool(payload.get("representation")),
        "parameter_matched_control": bool(payload.get("parameter_controls")),
        "oracle_graph_used": payload.get("oracle_graph_used") is False,
    }
    eligible = payload.get("scientific_status") == "real_model_mechanistic_evidence"
    return {
        "schema_version": 1,
        "decision": "GO" if eligible and all(criteria.values()) else "INCONCLUSIVE",
        "criteria": criteria,
        "scientific_status": payload.get("scientific_status"),
        "interpretation": (
            "Toy/fixture mechanisms are implementation evidence only."
            if not eligible
            else "All registered mechanistic criteria passed."
        ),
    }


def _load(value: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    payload = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("mechanistic suite must be a mapping")
    return payload


def _write_markdown(path: Path, headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_latex(
    path: Path,
    caption: str,
    headers: tuple[str, ...],
    rows: list[tuple[Any, ...]],
) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\begin{tabular}{lllll}",
        "\\toprule",
    ]
    lines.append(" & ".join(headers) + " \\\\ ")
    lines.append("\\midrule")
    lines.extend(
        " & ".join(str(value).replace("_", "\\_") for value in row) + " \\\\ "
        for row in rows
    )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _write_svg(path: Path) -> None:
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="360">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="32" y="72" font-family="sans-serif" font-size="26">'
        "Causal structure-token interventions</text>"
        '<text x="32" y="130" font-family="sans-serif" fill="#4b5563">'
        "Generated from paired remove / shuffle / replace outputs</text></svg>\n",
        encoding="utf-8",
        newline="\n",
    )


__all__ = ["generate_mechanistic_reports"]
