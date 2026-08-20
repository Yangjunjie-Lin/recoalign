"""Non-selective aggregation and paper artifact generation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluation.statistics import multiple_seed_summary, paired_bootstrap_test
from recoalign.reproducibility import atomic_write_json, utc_now

from ..analysis import (
    capability_preservation,
    compare_failure_taxonomy,
    mechanism_consistency,
)
from ..matrix import build_evaluation_matrix, load_matrix_config


def collect_cell_reports(results_root: str | Path) -> list[dict[str, Any]]:
    root = Path(results_root)
    rows: list[dict[str, Any]] = []
    for report_path in sorted(root.rglob("report.json")) if root.exists() else []:
        report = _load_json(report_path)
        metrics = _load_json(report_path.parent / "metrics.json")
        rows.append(
            {
                "path": report_path.parent.as_posix(),
                "report": report,
                "metrics": metrics,
                "model": report.get("model") or metrics.get("model"),
                "method": report.get("method") or metrics.get("method"),
                "benchmark": report.get("benchmark") or metrics.get("benchmark"),
                "seed": report.get("seed") or metrics.get("seed"),
                "status": report.get("status", metrics.get("status", "unknown")),
            }
        )
    return rows


def generate_comprehensive_reports(
    *,
    matrix_config: str | Path | dict[str, Any] = "configs/benchmarks/comprehensive_matrix.yaml",
    results_root: str | Path = "outputs/comprehensive",
    output_root: str | Path = "reports/comprehensive",
) -> dict[str, Any]:
    config = (
        load_matrix_config(matrix_config)
        if isinstance(matrix_config, (str, Path))
        else dict(matrix_config)
    )
    plan = build_evaluation_matrix(config)
    rows = collect_cell_reports(results_root)
    complete = [row for row in rows if row["status"] == "complete"]
    statistics = _aggregate(complete)
    prediction_analyses = _prediction_analyses(complete, config)
    diagnosis = _load_json(Path("outputs/EXP004/analysis.json"))
    mechanism_rows = [
        {
            "model": row["model"],
            "benchmark": row["benchmark"],
            "gain": row["recoalign_minus_original"]["statistic"],
        }
        for row in statistics["paired_comparisons"].values()
    ]
    mechanism = mechanism_consistency(diagnosis, mechanism_rows)
    decision = _decision(plan, complete, statistics, config)
    destination = Path(output_root)
    tables = destination / "tables"
    figures = destination / "figures"
    stats_dir = destination / "statistics"
    latex = destination / "latex"
    for directory in (tables, figures, stats_dir, latex):
        directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination / "matrix_plan.json", plan)
    atomic_write_json(destination / "cell_reports.json", rows)
    atomic_write_json(stats_dir / "multi_seed_statistics.json", statistics)
    atomic_write_json(stats_dir / "capability_preservation.json", prediction_analyses["capability"])
    atomic_write_json(stats_dir / "failure_analysis.json", prediction_analyses["failures"])
    atomic_write_json(stats_dir / "mechanism_consistency.json", mechanism)
    atomic_write_json(destination / "decision_report.json", decision)
    _write_tables(tables, latex, plan, rows, statistics)
    _write_figures(figures, statistics, complete)
    report = _markdown_report(plan, rows, statistics, decision)
    (destination / "comprehensive_evaluation_report.md").write_text(
        report, encoding="utf-8", newline="\n"
    )
    (destination / "benchmark_evaluation_report.md").write_text(
        report, encoding="utf-8", newline="\n"
    )
    (destination / "ablation_report.md").write_text(
        _ablation_report(plan, statistics), encoding="utf-8", newline="\n"
    )
    (destination / "mechanism_consistency_report.md").write_text(
        _mechanism_report(mechanism, prediction_analyses), encoding="utf-8", newline="\n"
    )
    return {
        "output_root": destination.as_posix(),
        "matrix_cells": plan["minimum_cells"],
        "observed_reports": len(rows),
        "complete_cells": len(complete),
        "decision": decision["decision"],
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["method"]), str(row["benchmark"]))].append(row)
    summaries: dict[str, Any] = {}
    for key, cells in sorted(grouped.items()):
        values = [_metric(cell["metrics"], "accuracy") for cell in cells]
        values = [value for value in values if value is not None]
        if not values:
            continue
        model, method, benchmark = key
        summaries[_key(*key)] = {
            "model": model,
            "method": method,
            "benchmark": benchmark,
            "seed_count": len(values),
            "accuracy": multiple_seed_summary(
                values,
                bootstrap_samples=1000,
                seed=17,
                significance_test="paired_bootstrap" if len(values) >= 3 else "paired_t_test",
            ),
            "failure_rate": 1.0 - sum(values) / len(values),
            "compositional": _metric(cells[0]["metrics"], "compositional"),
            "distribution": _metric(cells[0]["metrics"], "distribution"),
        }
    comparisons: dict[str, Any] = {}
    by_benchmark_model: dict[tuple[str, str], dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        accuracy = _metric(row["metrics"], "accuracy")
        if accuracy is not None:
            by_benchmark_model[(str(row["benchmark"]), str(row["model"]))][
                str(row["method"])
            ][int(row["seed"])] = accuracy
    for (benchmark, model), methods in sorted(by_benchmark_model.items()):
        if "recoalign" not in methods or "original_vlm" not in methods:
            continue
        common = sorted(set(methods["recoalign"]) & set(methods["original_vlm"]))
        if len(common) < 1:
            continue
        first = [methods["recoalign"][seed] for seed in common]
        second = [methods["original_vlm"][seed] for seed in common]
        comparisons[_key(model, benchmark)] = {
            "model": model,
            "benchmark": benchmark,
            "paired_seeds": common,
            "recoalign_minus_original": paired_bootstrap_test(
                first, second, samples=1000, seed=29
            ),
        }
    return {"cell_summaries": summaries, "paired_comparisons": comparisons}


def _decision(
    plan: dict[str, Any],
    complete: list[dict[str, Any]],
    statistics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    summary = statistics["cell_summaries"]
    recoalign = [row for row in summary.values() if row["method"] == "recoalign"]
    models = {row["model"] for row in recoalign}
    benchmarks = {row["benchmark"] for row in recoalign}
    paired = statistics["paired_comparisons"]
    positive = [
        row["recoalign_minus_original"]["statistic"] > 0
        for row in paired.values()
        if row["paired_seeds"]
    ]
    reasons: list[str] = []
    if len(models) < 2:
        reasons.append("fewer than two models have complete ReCoAlign evidence")
    if len(benchmarks) < 3:
        reasons.append("fewer than three benchmarks have complete ReCoAlign evidence")
    minimum_seeds = int(config.get("statistics", {}).get("minimum_seeds", 3))
    if any(row["seed_count"] < minimum_seeds for row in recoalign):
        reasons.append("one or more ReCoAlign cells have fewer than the registered seed count")
    if not positive or not all(positive):
        reasons.append("paired ReCoAlign improvements are absent or inconsistent")
    status = "GO" if not reasons and complete else "INCONCLUSIVE"
    return {
        "schema_version": 1,
        "decision": status,
        "criteria": {
            "multiple_models": len(models) >= 2,
            "multiple_benchmarks": len(benchmarks) >= 3,
            "minimum_seeds": minimum_seeds,
            "consistent_paired_gain": bool(positive) and all(positive),
            "all_matrix_cells_reported": len(complete) == plan["minimum_cells"],
            "no_selective_reporting": True,
        },
        "blockers": reasons,
        "complete_cells": len(complete),
        "planned_cells": plan["minimum_cells"],
        "scientific_claim": (
            "ReCoAlign comprehensive efficacy is supported"
            if status == "GO"
            else "No comprehensive scientific claim is made until the frozen matrix is complete"
        ),
        "generated_at": utc_now(),
    }


def _prediction_analyses(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    pairs: dict[tuple[str, str, int], dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    categories = {
        str(row["name"]): str(row["category"]) for row in config.get("benchmarks", [])
    }
    for row in rows:
        method = str(row["method"])
        if method not in {"original_vlm", "recoalign"}:
            continue
        key = (str(row["model"]), str(row["benchmark"]), int(row["seed"]))
        pairs[key][method] = _load_jsonl(Path(row["path"]) / "predictions.jsonl")
    capability: dict[str, Any] = {}
    failures: dict[str, Any] = {}
    threshold = float(config.get("capability_preservation", {}).get("maximum_degradation", 0.02))
    for key, methods in sorted(pairs.items()):
        if not {"original_vlm", "recoalign"} <= set(methods):
            continue
        identifier = _key(*key)
        failures[identifier] = compare_failure_taxonomy(
            methods["original_vlm"], methods["recoalign"]
        )
        if categories.get(key[1]) == "general_reasoning":
            capability[identifier] = capability_preservation(
                methods["original_vlm"],
                methods["recoalign"],
                maximum_degradation=threshold,
            )
    return {"capability": capability, "failures": failures}


def _write_tables(
    tables: Path,
    latex: Path,
    plan: dict[str, Any],
    rows: list[dict[str, Any]],
    statistics: dict[str, Any],
) -> None:
    headers = ("Model", "Benchmark", "Method", "Seeds", "Accuracy", "Status")
    all_rows: list[tuple[Any, ...]] = []
    lookup = {(row["model"], row["benchmark"], row["method"]): row for row in rows}
    seen: set[tuple[str, str, str]] = set()
    for cell in plan["cells"]:
        key = (cell["model"], cell["benchmark"], cell["method"])
        if key in seen:
            continue
        seen.add(key)
        observed = lookup.get(key)
        if observed:
            cell_summary = statistics["cell_summaries"].get(_key(*key), {})
            accuracy = cell_summary.get("accuracy", {}).get("mean", "")
            status = observed.get("status", "unknown")
            seeds = cell_summary.get("seed_count", 0)
        else:
            accuracy, status, seeds = "", cell["status"], 0
        all_rows.append((cell["model"], cell["benchmark"], cell["method"], seeds, accuracy, status))
    _markdown_table(tables / "table_main_results.md", headers, all_rows)
    _latex_table(latex / "table_main_results.tex", "Main results", headers, all_rows)
    ablations = [row for row in all_rows if row[2] not in {"original_vlm", "recoalign"}]
    _markdown_table(tables / "table_ablation.md", headers, ablations)
    _latex_table(latex / "table_ablation.tex", "Ablation results", headers, ablations)
    generalization = [row for row in all_rows if row[1] in {"gqa", "mmvp", "crepe"}]
    _markdown_table(tables / "table_generalization.md", headers, generalization)
    _latex_table(
        latex / "table_generalization.tex",
        "Generalization results",
        headers,
        generalization,
    )


def _write_figures(figures: Path, statistics: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    _write_svg(
        figures / "figure_1_framework.svg",
        "ReCoAlign evaluation framework",
        "visual semantics → structure interface → reasoning",
    )
    _write_svg(
        figures / "figure_2_mechanism.svg",
        "Mechanism validation",
        "Graph / Caption / Image conditions — pending matrix evidence",
    )
    _write_svg(
        figures / "figure_3_multimodel.svg",
        "Multi-model comparison",
        _model_summary(statistics),
    )
    _write_svg(
        figures / "figure_4_ood.svg",
        "IID versus OOD generalization",
        _ood_summary(statistics),
    )
    _write_svg(figures / "figure_5_failure.svg", "Failure analysis", _failure_summary(rows))


def _markdown_report(
    plan: dict[str, Any],
    rows: list[dict[str, Any]],
    statistics: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    del statistics
    return "\n".join(
        [
            "# Comprehensive multi-VLM evaluation report",
            "",
            f"Decision: **{decision['decision']}**",
            "",
            "This report is generated from the complete registered matrix. Missing, blocked, "
            "dry-run, and failed cells remain visible in the tables; no selective result "
            "filtering is applied.",
            "",
            f"- Planned cells: {plan['minimum_cells']}",
            f"- Observed cell reports: {len(rows)}",
            f"- Complete inference cells: {decision['complete_cells']}",
            f"- Scientific blockers: {('; '.join(decision['blockers']) or 'none')}",
            "",
            "## Evidence scope",
            "",
            "The repository currently provides BaseVLM adapters and dry-run validation for "
            "LLaVA-1.5, LLaVA-NeXT, Qwen-VL, and InternVL. A ReCoAlign result is "
            "claim-eligible only when a trained interface checkpoint and backend hidden-context "
            "injection are both present. Dry-run and injected fixture results are infrastructure "
            "evidence, not paper claims.",
            "",
            "## Artifacts",
            "",
            "- `tables/table_main_results.md` and LaTeX counterpart",
            "- `tables/table_ablation.md` and LaTeX counterpart",
            "- `tables/table_generalization.md` and LaTeX counterpart",
            "- `figures/figure_1_framework.svg` through `figure_5_failure.svg`",
            "- `statistics/multi_seed_statistics.json`",
            "",
        ]
    )


def _ablation_report(plan: dict[str, Any], statistics: dict[str, Any]) -> str:
    methods = set(plan["methods"])
    ablations = methods - {"original_vlm", "caption_reasoning", "oracle_graph_prompt", "recoalign"}
    observed = {
        row["method"]
        for row in statistics["cell_summaries"].values()
        if row["method"] in ablations
    }
    return "\n".join(
        [
            "# Ablation report",
            "",
            f"Registered ablations: {', '.join(sorted(ablations)) or 'separate ablation matrix'}.",
            f"Completed ablations: {', '.join(sorted(observed)) or 'none'}.",
            "",
            "Architecture, training-loss, and data-source ablations use the same model, benchmark, "
            "seed, prompt, decoding, and evaluation protocol. Missing and failed cells are "
            "retained.",
            "",
        ]
    )


def _mechanism_report(mechanism: dict[str, Any], analyses: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Mechanism consistency report",
            "",
            f"Status: **{mechanism['scientific_status']}**",
            "",
            f"- Diagnosed interface gap: {mechanism['diagnosed_interface_gap']}",
            f"- Mechanism-consistent gains: {mechanism['mechanism_consistent']}",
            f"- Paired failure analyses: {len(analyses['failures'])}",
            f"- Capability-preservation analyses: {len(analyses['capability'])}",
            "",
            "A positive mechanism claim requires eligible EXP004 real-model diagnosis and paired "
            "ReCoAlign gains on the same model families. Infrastructure/reference outputs are not "
            "promoted to claim evidence.",
            "",
        ]
    )


def _markdown_table(path: Path, headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _latex_table(
    path: Path,
    caption: str,
    headers: tuple[str, ...],
    rows: list[tuple[Any, ...]],
) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\begin{tabular}{llllll}",
        "\\toprule",
    ]
    lines.append(" & ".join(headers) + " \\\\ ")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(str(value).replace("_", "\\_") for value in row) + " \\\\ ")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _write_svg(path: Path, title: str, subtitle: str) -> None:
    escaped_title = title.replace("&", "&amp;")
    escaped_subtitle = subtitle.replace("&", "&amp;")
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="420">'
        f'<rect width="100%" height="100%" fill="white"/><text x="40" y="80" '
        f'font-family="sans-serif" font-size="28">{escaped_title}</text><text x="40" y="140" '
        f'font-family="sans-serif" font-size="18" fill="#4b5563">{escaped_subtitle}</text>'
        '<rect x="40" y="200" width="820" height="120" fill="#f3f4f6" stroke="#9ca3af"/>'
        '<text x="60" y="270" font-family="sans-serif" font-size="18" fill="#6b7280">'
        'Complete frozen-matrix evidence required before scientific interpretation.</text></svg>\n',
        encoding="utf-8",
        newline="\n",
    )


def _model_summary(statistics: dict[str, Any]) -> str:
    models = sorted({row["model"] for row in statistics["cell_summaries"].values()})
    return ", ".join(models) or "all registered models — pending"


def _ood_summary(statistics: dict[str, Any]) -> str:
    return "OOD retention and generalization gap — pending complete IID/OOD cells"


def _failure_summary(rows: list[dict[str, Any]]) -> str:
    failures = sum(
        1
        for row in rows
        if row["metrics"].get("failure_rate") is not None
    )
    return f"failure taxonomy available for {failures} completed cells"


def _key(*values: Any) -> str:
    return "::".join(str(value) for value in values)


def _metric(metrics: dict[str, Any], key: str) -> Any:
    return metrics.get(key)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


__all__ = ["collect_cell_reports", "generate_comprehensive_reports"]
