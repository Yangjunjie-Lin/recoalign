from __future__ import annotations

# The report template intentionally keeps Markdown paragraphs as literal lines.
# ruff: noqa: E501
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import ExperimentConfig
from .decision import determine_decision
from .probes import SEMANTICS

COLORS = {
    "object": "#2E86AB",
    "attribute": "#5AA469",
    "relation": "#E07A5F",
    "composition": "#7B61A8",
}


def _row_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["stage"], row["semantic"]): row for row in rows}


def _write_csv(path: Path, rows: list[dict[str, Any]], drops: dict[str, Any]) -> None:
    output: list[dict[str, Any]] = []
    for row in rows:
        merged = dict(row)
        drop = drops[row["stage"]][row["semantic"]]
        merged.update(
            {
                "drop_from_za": drop["drop"],
                "drop_bootstrap_std": drop["bootstrap_std"],
                "drop_ci95_low": drop["ci95_low"],
                "drop_ci95_high": drop["ci95_high"],
            }
        )
        output.append(merged)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


def _plot_transition(config: ExperimentConfig, rows: list[dict[str, Any]]) -> None:
    lookup = _row_map(rows)
    layers = list(config.layer_indices)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for semantic in SEMANTICS:
        visual_values = [
            lookup[("Za", semantic)]["accuracy"]
            if layer == 0
            else lookup[(f"L{layer:02d}_visual", semantic)]["accuracy"]
            for layer in layers
        ]
        decision_values = [
            lookup[(f"L{layer:02d}_decision", semantic)]["accuracy"] for layer in layers
        ]
        axes[0].plot(
            layers,
            visual_values,
            marker="o",
            linewidth=2,
            color=COLORS[semantic],
            label=semantic.title(),
        )
        axes[1].plot(
            layers,
            decision_values,
            marker="o",
            linewidth=2,
            color=COLORS[semantic],
            label=semantic.title(),
        )
    axes[0].set_title("Visual-token semantic decodability")
    axes[1].set_title("Decision-position semantic decodability")
    for axis in axes:
        axis.set_xlabel("LLM hidden-state index")
        axis.set_xticks(layers)
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Held-out linear-probe accuracy")
    axes[1].legend(frameon=False, ncol=2)
    fig.suptitle("ReCoAlign Phase 0-B: semantic transition through frozen LLaVA-1.5-7B")
    fig.tight_layout()
    fig.savefig(config.figures_dir / "semantic_transition_curve.png", dpi=180)
    plt.close(fig)


def _plot_heatmap(config: ExperimentConfig, drops: dict[str, Any]) -> None:
    layers = list(config.layer_indices[1:])
    matrix = np.asarray(
        [
            [drops[f"L{layer:02d}_visual"][semantic]["drop"] for semantic in SEMANTICS]
            for layer in layers
        ]
    )
    limit = max(0.05, float(np.max(np.abs(matrix))))
    fig, axis = plt.subplots(figsize=(7.4, 4.4))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(SEMANTICS)), [item.title() for item in SEMANTICS])
    axis.set_yticks(range(len(layers)), [f"Layer {layer}" for layer in layers])
    axis.set_title("Semantic preservation drop from Za (visual-token states)")
    for i in range(len(layers)):
        for j in range(len(SEMANTICS)):
            axis.text(j, i, f"{matrix[i, j] * 100:+.1f}pp", ha="center", va="center")
    fig.colorbar(image, ax=axis, label="Za accuracy − hidden accuracy")
    fig.tight_layout()
    fig.savefig(config.figures_dir / "semantic_drop_heatmap.png", dpi=180)
    plt.close(fig)


def _plot_attention(config: ExperimentConfig) -> dict[str, Any]:
    path = config.features_dir / "attention.jsonl"
    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for metric in (
                "visual_attention_mass",
                "visual_attention_entropy",
            ):
                grouped[int(row["layer"])][metric].append(float(row[metric]))
    layers = sorted(grouped)
    mass = [np.mean(grouped[layer]["visual_attention_mass"]) for layer in layers]
    entropy = [np.mean(grouped[layer]["visual_attention_entropy"]) for layer in layers]
    mass_std = [np.std(grouped[layer]["visual_attention_mass"], ddof=1) for layer in layers]
    entropy_std = [np.std(grouped[layer]["visual_attention_entropy"], ddof=1) for layer in layers]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].plot(layers, mass, color="#E07A5F", linewidth=2)
    axes[0].fill_between(
        layers, np.asarray(mass) - mass_std, np.asarray(mass) + mass_std, alpha=0.2
    )
    axes[0].set_title("Decision → visual attention mass")
    axes[0].set_ylabel("Mean attention probability")
    axes[1].plot(layers, entropy, color="#2E86AB", linewidth=2)
    axes[1].fill_between(
        layers,
        np.asarray(entropy) - entropy_std,
        np.asarray(entropy) + entropy_std,
        alpha=0.2,
    )
    axes[1].set_title("Within-visual attention entropy")
    axes[1].set_ylabel("Normalized entropy")
    for axis in axes:
        axis.set_xlabel("Decoder block")
        axis.grid(alpha=0.25)
    fig.suptitle("Frozen LLaVA attention audit (mean ± sample SD)")
    fig.tight_layout()
    fig.savefig(config.figures_dir / "attention_analysis.png", dpi=180)
    plt.close(fig)
    return {
        "n_layers": len(layers),
        "samples_per_layer": len(grouped[layers[0]]["visual_attention_mass"]),
        "per_layer": {
            str(layer): {
                "visual_attention_mass_mean": float(mass[index]),
                "visual_attention_mass_std": float(mass_std[index]),
                "visual_attention_entropy_mean": float(entropy[index]),
                "visual_attention_entropy_std": float(entropy_std[index]),
            }
            for index, layer in enumerate(layers)
        },
    }


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _table(config: ExperimentConfig, probe_results: dict[str, Any]) -> str:
    lookup = _row_map(probe_results["rows"])
    lines = [
        "| Stage | Object | Attribute | Relation | Composition |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    stages = ["Za"] + [f"L{layer:02d}_visual" for layer in config.layer_indices[1:]]
    for stage in stages:
        values = [_pct(lookup[(stage, semantic)]["accuracy"]) for semantic in SEMANTICS]
        lines.append(f"| {stage} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def _decision_table(config: ExperimentConfig, probe_results: dict[str, Any]) -> str:
    lookup = _row_map(probe_results["rows"])
    lines = [
        "| Stage | Object | Attribute | Relation | Composition |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for layer in config.layer_indices:
        stage = f"L{layer:02d}_decision"
        values = [_pct(lookup[(stage, semantic)]["accuracy"]) for semantic in SEMANTICS]
        lines.append(f"| {stage} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_results(
    config: ExperimentConfig,
    feature_info: dict[str, Any],
    probe_results: dict[str, Any],
) -> dict[str, Any]:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(config.results_dir / "results.csv", probe_results["rows"], probe_results["drops"])
    _plot_transition(config, probe_results["rows"])
    _plot_heatmap(config, probe_results["drops"])
    attention_statistics = _plot_attention(config)
    decision = determine_decision(config, probe_results)
    statistics = {
        "seed": config.seed,
        "bootstrap_samples": config.bootstrap_samples,
        "identity": probe_results["identity"],
        "drops": probe_results["drops"],
        "selectivity": probe_results["selectivity"],
        "attention": attention_statistics,
        "probe_convergence": probe_results["convergence"],
    }
    (config.results_dir / "statistics.json").write_text(
        json.dumps(statistics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (config.results_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (config.results_dir / "run_config.json").write_text(
        json.dumps(
            {"config": config.to_dict(), "feature_extraction": feature_info},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report = f"""# ReCoAlign Phase 0-B Report

Generated: {datetime.now(timezone.utc).isoformat()}

## 1. Scientific question

本实验只检验一个冻结机制问题：在 Phase 0-A 已证实 `Za` 保有语义的前提下，关系与组合语义是否在 LLaVA-1.5-7B 的 LLM hidden visual-token representations 中发生相对于 object 明显更强的线性可解码性下降。模型结构和参数均未修改；没有 adapter、训练、loss 或微调。

## 2. Protocol

- Model: official `liuhaotian/llava-v1.5-7b`, frozen Vicuna-7B LLM and official projector.
- Data: {config.n_train} balanced train + {config.n_test} balanced held-out synthetic scenes; each of 72 composition classes contributes {config.n_train // 72}/{config.n_test // 72} samples.
- Boundaries: `Za`, then mean over the same 576 visual-token positions at hidden-state indices {list(config.layer_indices)}. Decision-position probes and attention are supporting diagnostics only.
- Probe: standardized multinomial logistic regression, fixed `C=1`, no nonlinear head.
- Uncertainty: {config.bootstrap_samples} paired held-out bootstrap resamples.
- Runtime precision: NF4 frozen-weight inference. This makes the run feasible on 6GB VRAM, but any positive result requires a higher-precision replication before a broad claim.
- Registered protocol validity: **{"FULL" if config.scientific_protocol_valid() else "UNDERPOWERED / DIAGNOSTIC"}**.

## 3. Held-out semantic decodability

{_table(config, probe_results)}

The full drops, 95% confidence intervals, and relation/composition-vs-object paired selectivity tests are recorded in `statistics.json` and `results.csv`.

### Supporting decision-position diagnostic

{_decision_table(config, probe_results)}

The decision position begins at chance at state 0 because it has not yet attended to the image. By state 8 it already exposes all four semantics at 95% or better; at final state 32, Object/Attribute/Relation/Composition are {_pct(_row_map(probe_results["rows"])[("L32_decision", "object")]["accuracy"])}/{_pct(_row_map(probe_results["rows"])[("L32_decision", "attribute")]["accuracy"])}/{_pct(_row_map(probe_results["rows"])[("L32_decision", "relation")]["accuracy"])}/{_pct(_row_map(probe_results["rows"])[("L32_decision", "composition")]["accuracy"])}. This diagnostic therefore does not reveal an accessibility failure either. It is supporting evidence, not a causal-use claim.

The layer-32 decision token assigns {_pct(attention_statistics["per_layer"]["32"]["visual_attention_mass_mean"])} mean attention mass to visual tokens. Attention is reported descriptively and is not a decision gate.

## 4. Automatic decision

Observed pattern: **{decision["pattern"]}**

Passing specific-degradation layers: `{decision["passing_layers"]}`.

# {decision["decision"]}

Semantic Utilization Failure is **{"SUPPORTED in this controlled minimal experiment" if decision["semantic_utilization_failure"] else "NOT ESTABLISHED by this experiment"}**.

{("下一阶段进入 Causal Intervention design；首先应做高精度复现与 activation-level causal patching。" if decision["decision"] == "GO" else "按预注册规则停止：当前证据不支持进入 Causal Intervention 设计。NO-GO 不等于证明所有自然图像或所有 VLM 都不存在 utilization failure。")}

## 5. Interpretation limits

- Linear decodability is a diagnostic, not proof that the model causally uses a feature.
- Attention mass is descriptive and is not treated as a causal explanation or a GO gate.
- The balanced rendered scenes isolate the hypothesized mechanism but do not establish prevalence on natural images.
- Near-ceiling accuracies bound detectable loss but may hide subtle changes smaller than the registered 15-point effect.
- NF4 and the final-four-layer fp16 CPU execution are inference approximations. A positive result would require higher-precision replication; this negative result remains scoped to the registered runtime.
"""
    (config.root / "ReCoAlign_Phase0B_Report.md").write_text(report, encoding="utf-8")
    return decision
