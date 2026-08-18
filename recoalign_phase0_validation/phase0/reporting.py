# The long lines in this module are Markdown paragraphs in generated reports.
# ruff: noqa: E501

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .config import ExperimentConfig
from .probes import SEMANTICS, STAGES


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def determine_decision(config: ExperimentConfig, probe_results: dict[str, Any]) -> dict[str, Any]:
    drops = probe_results["drops"]
    comparison = probe_results["object_vs_relation"]
    isolation = probe_results["semantic_isolation_drop"]
    pattern_a = (
        drops["object"]["drop"] < 0.10
        and drops["attribute"]["drop"] < 0.15
        and drops["relation"]["drop"] > 0.25
        and drops["composition"]["drop"] > 0.25
    )
    comparison_significant = comparison["p_one_sided"] < 0.05
    isolation_corroborates = isolation["drop"] > 0.0 and isolation["ci95_low"] > 0.0
    protocol_valid = config.scientific_protocol_valid()
    go = protocol_valid and pattern_a and comparison_significant and isolation_corroborates

    drop_values = np.asarray([drops[name]["drop"] for name in SEMANTICS], dtype=np.float64)
    if pattern_a:
        pattern = "Pattern A: selective semantic degradation"
    elif float(np.max(drop_values)) < 0.05:
        pattern = "Pattern C: no material degradation at the tested projection"
    elif float(np.min(drop_values)) > 0.10 and float(np.ptp(drop_values)) < 0.10:
        pattern = "Pattern B: approximately uniform degradation (generic bottleneck)"
    else:
        pattern = "Mixed/inconclusive pattern"
    if not protocol_valid:
        pattern += "; run is below the pre-registered full protocol"

    return {
        "decision": "GO" if go else "NO-GO",
        "pattern": pattern,
        "protocol_valid": protocol_valid,
        "pattern_a_thresholds_met": pattern_a,
        "object_vs_relation_significant": comparison_significant,
        "semantic_isolation_corroborates": isolation_corroborates,
        "gates": {
            "object_drop_lt_10pp": drops["object"]["drop"] < 0.10,
            "attribute_drop_lt_15pp": drops["attribute"]["drop"] < 0.15,
            "relation_drop_gt_25pp": drops["relation"]["drop"] > 0.25,
            "composition_drop_gt_25pp": drops["composition"]["drop"] > 0.25,
            "relation_vs_object_p_lt_0_05": comparison_significant,
            "isolation_drop_ci_above_zero": isolation_corroborates,
            "full_protocol": protocol_valid,
        },
    }


def _main_row(rows: list[dict[str, Any]], semantic: str, stage: str) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if row["evaluation"] == "main" and row["semantic"] == semantic and row["stage"] == stage
    )


def _isolation_row(rows: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    return next(
        row for row in rows if row["evaluation"] == "semantic_isolation" and row["stage"] == stage
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = (
        "evaluation",
        "semantic",
        "stage",
        "accuracy",
        "bootstrap_mean",
        "bootstrap_std",
        "ci95_low",
        "ci95_high",
        "drop_from_zv",
        "n_train",
        "n_test",
        "num_classes",
        "pair_accuracy",
        "converged",
        "iterations",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in columns})


def _plot_figures(output_dir: Path, probe_results: dict[str, Any]) -> None:
    rows = probe_results["rows"]
    drops = probe_results["drops"]
    sns.set_theme(style="whitegrid", context="talk")
    colors = sns.color_palette("colorblind", n_colors=len(SEMANTICS))

    fig, ax = plt.subplots(figsize=(9, 6))
    for color, semantic in zip(colors, SEMANTICS, strict=True):
        values = [_main_row(rows, semantic, stage)["accuracy"] for stage in STAGES]
        ax.plot(
            STAGES,
            values,
            marker="o",
            linewidth=2.5,
            markersize=8,
            label=semantic.title(),
            color=color,
        )
    ax.set_xlabel("Representation stage")
    ax.set_ylabel("Linear-probe accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Semantic preservation curve")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "figure1_semantic_preservation_curve.png", dpi=180)
    plt.close(fig)

    heat_values = np.asarray([[100.0 * drops[name]["drop"] for name in SEMANTICS]])
    fig, ax = plt.subplots(figsize=(10, 2.8))
    sns.heatmap(
        heat_values,
        annot=True,
        fmt=".1f",
        cmap="vlag",
        center=0.0,
        xticklabels=[name.title() for name in SEMANTICS],
        yticklabels=["Zv − Za (pp)"],
        cbar_kws={"label": "Accuracy drop (percentage points)"},
        ax=ax,
    )
    ax.set_title("Semantic degradation heatmap")
    fig.tight_layout()
    fig.savefig(output_dir / "figure2_semantic_degradation_heatmap.png", dpi=180)
    plt.close(fig)

    means = [100.0 * drops[name]["drop"] for name in SEMANTICS]
    lows = [100.0 * drops[name]["ci95_low"] for name in SEMANTICS]
    highs = [100.0 * drops[name]["ci95_high"] for name in SEMANTICS]
    errors = np.asarray(
        [
            [mean - low for mean, low in zip(means, lows, strict=True)],
            [high - mean for mean, high in zip(means, highs, strict=True)],
        ]
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(SEMANTICS))
    bars = ax.bar(x, means, yerr=errors, capsize=6, color=colors)
    for bar, value in zip(bars, means, strict=True):
        offset = 0.35 if value >= 0 else -0.8
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=10,
        )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axhline(25.0, color="gray", linewidth=1.5, linestyle="--", label="25 pp selective-loss gate")
    ax.set_xticks(x, [name.title() for name in SEMANTICS])
    ax.set_ylabel("SPD = Accuracy(Zv) − Accuracy(Za), pp")
    ax.set_title("Semantic factor comparison (95% bootstrap CI)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "figure3_semantic_factor_comparison.png", dpi=180)
    plt.close(fig)


def _accuracy_table(rows: list[dict[str, Any]], drops: dict[str, Any]) -> str:
    lines = [
        "| Semantic | Zv accuracy | Za accuracy | SPD (Zv − Za) | 95% CI of drop |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for semantic in SEMANTICS:
        zv = _main_row(rows, semantic, "Zv")["accuracy"]
        za = _main_row(rows, semantic, "Za")["accuracy"]
        drop = drops[semantic]
        lines.append(
            f"| {semantic.title()} | {_pct(zv)} | {_pct(za)} | {_pct(drop['drop'])} | "
            f"[{_pct(drop['ci95_low'])}, {_pct(drop['ci95_high'])}] |"
        )
    return "\n".join(lines)


def write_results(
    config: ExperimentConfig,
    feature_info: dict[str, Any],
    probe_results: dict[str, Any],
) -> dict[str, Any]:
    output_dir = config.results_dir
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "results.csv", probe_results["rows"])
    _plot_figures(figures_dir, probe_results)

    decision = determine_decision(config, probe_results)
    (output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {"config": config.to_dict(), "feature_extraction": feature_info},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = probe_results["rows"]
    drops = probe_results["drops"]
    comparison = probe_results["object_vs_relation"]
    isolation = probe_results["semantic_isolation_drop"]
    iso_zv = _isolation_row(rows, "Zv")
    iso_za = _isolation_row(rows, "Za")
    gate_lines = "\n".join(
        f"- `{name}`: {'PASS' if passed else 'FAIL'}" for name, passed in decision["gates"].items()
    )
    report = f"""# ReCoAlign Phase 0 Report

Generated: {datetime.now(timezone.utc).isoformat()}

## 1. Experimental Setup

This run tests whether a frozen cross-modal projection selectively removes semantic structure that is linearly decodable from its frozen vision input. The dataset contains {config.n_train + config.n_test:,} main images ({config.n_train:,} train / {config.n_test:,} test) and {config.n_control_pairs:,} relation-isolation pairs. The frozen backend is `{feature_info["backend"]}` using `{feature_info["model_name"]}`. `Zv` and `Za` are captured at the representation boundaries documented in `run_config.json`; neither the encoder nor the projection is trained or modified. Only logistic-regression probes are fitted.

All randomness uses seed `{config.seed}`. Accuracy uncertainty is estimated with {config.bootstrap_samples:,} paired bootstrap resamples. Standard deviations and confidence intervals are in `results.csv`; per-sample predictions are retained under `predictions/`.

Protocol validity: **{"FULL" if decision["protocol_valid"] else "UNDERPOWERED / DIAGNOSTIC ONLY"}**.

## 2. Results

{_accuracy_table(rows, drops)}

The paired bootstrap estimate for `Relation drop − Object drop` is {_pct(comparison["observed_difference"])}, 95% CI [{_pct(comparison["ci95_low"])}, {_pct(comparison["ci95_high"])}], one-sided p = {comparison["p_one_sided"]:.6f}.

Semantic-isolation relation accuracy is {_pct(iso_zv["accuracy"])} at Zv and {_pct(iso_za["accuracy"])} at Za (drop {_pct(isolation["drop"])}, 95% CI [{_pct(isolation["ci95_low"])}, {_pct(isolation["ci95_high"])}]). Pair-level both-correct accuracy is {_pct(iso_zv["pair_accuracy"])} at Zv and {_pct(iso_za["pair_accuracy"])} at Za.

## 3. Evidence for Semantic Preservation Failure

Observed pattern: **{decision["pattern"]}**.

Pre-registered decision gates:

{gate_lines}

The experiment calls selective preservation failure only when the exact asymmetric thresholds are met, the Relation-vs-Object drop is significant, the relation-isolation control has a drop whose 95% CI excludes zero, and the full protocol was run.

## 4. Alternative Explanation Analysis

- **Relation is intrinsically harder:** absolute Relation accuracy may be lower, but SPD compares the same task before and after projection. The paired isolation set strengthens this control by holding objects, attributes, rendering nuisance, and pair identity fixed while changing only the relation.
- **Generic alignment bottleneck:** similar drops across all four factors are classified as Pattern B and produce NO-GO. Only the asymmetric Pattern A can pass.
- **Probe learns the task:** every probe is a single logistic-regression layer with fixed regularization; no nonlinear probe, VLM fine-tuning, or learned representation is used.
- **Ceiling effect / synthetic task too easy:** near-perfect accuracy at both stages limits the experiment's ability to expose subtle selective loss. This is a reason to retain NO-GO rather than reinterpret a null result as proof that all VLM semantics are preserved.
- **Synthetic-world scope:** a positive result would establish a controlled mechanism, not prevalence in natural imagery. A negative result only rejects this mechanism for the tested projection and synthetic distribution; it does not prove all VLM stages preserve semantics.
- **LLM-stage scope:** this Phase 0 path intentionally stops after the alignment projector and does not load the 7B language model, so `Zl` is unavailable. The run tests the registered `Zv → Za` mechanism only; it makes no claim about losses inside LLM layers.

## 5. Final Decision

# {decision["decision"]}

{("The evidence satisfies every pre-registered gate for selective semantic degradation, so a bounded LLaVA replication is warranted." if decision["decision"] == "GO" else "The experiment does not satisfy every pre-registered selective-degradation gate. ReCoAlign is not advanced on the basis of this run; no favorable post-hoc interpretation is substituted.")}
"""
    (output_dir / "ReCoAlign_Phase0_Report.md").write_text(report, encoding="utf-8")

    if feature_info["backend"] == "llava_1_5_7b_vision_projector":
        reproduce_command = (
            f"python run.py --backend llava --device cuda --batch-size {config.batch_size}"
        )
    else:
        reproduce_command = (
            f"python run.py --backend clip --device {feature_info['device']} "
            f"--batch-size {config.batch_size}"
        )
    result_readme = f"""# Experiment Results

This directory was generated by `python run.py` and is tied to the exact configuration in `run_config.json`.

Final decision: **{decision["decision"]}**

Observed classification: {decision["pattern"]}.

Artifacts:

- `results.csv`: all main and semantic-isolation probe accuracies, bootstrap standard deviations, 95% confidence intervals, and SPD values;
- `statistics.json`: paired bootstrap drops and Object-vs-Relation significance test;
- `decision.json`: machine-readable gates and final GO / NO-GO;
- `ReCoAlign_Phase0_Report.md`: complete automatic report;
- `figures/`: the three requested figures;
- `predictions/`: per-sample labels and predictions for auditability.

Re-run from the repository directory with:

```bash
{reproduce_command}
```
"""
    (output_dir / "README.md").write_text(result_readme, encoding="utf-8")
    return decision
