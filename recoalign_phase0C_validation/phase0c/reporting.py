from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import atomic_json


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _plot_semantic_gap(
    path: Path, availability_rows: list[dict[str, Any]], behavior_accuracy: float
) -> None:
    layers = [row["layer"] for row in availability_rows]
    availability = [row["accuracy"] for row in availability_rows]
    low = [row["accuracy"] - row["accuracy_ci_low"] for row in availability_rows]
    high = [row["accuracy_ci_high"] - row["accuracy"] for row in availability_rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.errorbar(
        layers,
        availability,
        yerr=np.array([low, high]),
        color="#1768AC",
        marker="o",
        linewidth=2.2,
        capsize=4,
        label="Representation availability (probe)",
    )
    ax.axhline(
        behavior_accuracy,
        color="#D1495B",
        linewidth=2.2,
        linestyle="--",
        label="Behavioral utilization (frozen LM head)",
    )
    ax.axhline(0.25, color="#777777", linewidth=1.2, linestyle=":", label="Chance")
    ax.set(xlabel="Decision-position hidden state", ylabel="Held-out accuracy", ylim=(0, 1.04))
    ax.set_xticks(layers, [f"L{layer:02d}" for layer in layers])
    ax.set_title("Representation availability versus behavioral utilization")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_attention(path: Path, attention_statistics: dict[str, Any]) -> None:
    layers = sorted(int(key) for key in attention_statistics["per_layer"])
    correct = [attention_statistics["per_layer"][str(layer)]["correct"]["mean"] for layer in layers]
    incorrect = [
        attention_statistics["per_layer"][str(layer)]["incorrect"]["mean"] for layer in layers
    ]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.plot(layers, correct, color="#2A9D8F", linewidth=2, label="Behavior correct")
    if all(value is not None for value in incorrect):
        ax.plot(layers, incorrect, color="#E76F51", linewidth=2, label="Behavior incorrect")
    ax.set(
        xlabel="Decoder layer",
        ylabel="Mean attention mass to 576 visual tokens",
        title="Decision-to-vision attention (supporting diagnostic)",
    )
    ax.grid(alpha=0.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_recovery(path: Path, config: Any, patch_statistics: dict[str, Any]) -> None:
    alphas = list(config.patch_alphas)
    curve = patch_statistics.get("curve", {})
    baseline_key = f"L{config.primary_patch_layer:02d}|0.00|baseline"
    baseline = curve.get(baseline_key, {}).get("mean")
    positive: list[float] = []
    control: list[float] = []
    for alpha in alphas:
        if alpha == 0:
            positive.append(float(baseline or 0.0))
            control.append(float(baseline or 0.0))
        else:
            positive.append(
                float(
                    curve.get(
                        f"L{config.primary_patch_layer:02d}|{alpha:.2f}|positive", {}
                    ).get("mean")
                    or 0.0
                )
            )
            control.append(
                float(
                    curve.get(
                        f"L{config.primary_patch_layer:02d}|{alpha:.2f}|control", {}
                    ).get("mean")
                    or 0.0
                )
            )
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(alphas, positive, marker="o", linewidth=2.2, color="#1768AC", label="Positive donor")
    ax.plot(alphas, control, marker="s", linewidth=2.2, color="#B56576", label="Counterfactual control donor")
    ax.set(
        xlabel="Donor activation blend α",
        ylabel="Recovery rate on baseline failures",
        title=f"Layer-{config.primary_patch_layer} activation-patch recovery",
        ylim=(-0.03, 1.03),
    )
    ax.set_xticks(alphas)
    ax.grid(axis="y", alpha=0.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _format_percent(value: Any) -> str:
    return "N/A" if value is None else f"{100 * float(value):.2f}%"


def _write_report(
    path: Path,
    config: Any,
    decision: dict[str, Any],
    availability_rows: list[dict[str, Any]],
    statistics: dict[str, Any],
) -> None:
    primary = next(row for row in availability_rows if row["layer"] == 32)
    patch = statistics["activation_patch"]["primary"]
    secondary_l24 = statistics["activation_patch"]["per_layer_full_patch"]["24"]
    rows = "\n".join(
        f"| L{row['layer']:02d} | {_format_percent(row['accuracy'])} | "
        f"{_format_percent(row['behavior_accuracy'])} | "
        f"{_format_percent(row['availability_behavior_gap'])} | "
        f"{_format_percent(row['error_subset_accuracy'])} |"
        for row in availability_rows
    )
    gates = "\n".join(
        f"- [{'x' if passed else ' '}] `{name}`" for name, passed in decision["gates"].items()
    )
    route = (
        "进入更高精度、自然图像的 Causal Intervention 复现实验。"
        if decision["decision"] == "GO"
        else "ReCoAlign 机制发现路线终止。"
    )
    text = f"""# ReCoAlign Phase 0-C Report

## 1. Scientific question

本实验检验 Semantic Access Failure：同一行为查询的决策位置中，空间关系信息是否可由 held-out 线性 probe 解码，却未被冻结模型原有 LM head 用于同一个四分类决定。

本报告只对以下范围作结论：official LLaVA-1.5-7B、NF4 冻结推理、固定合成图像、`left/right/above/below` 下一 token 决策。它不是对所有 Vision-Language Models 的普遍性证明。

## 2. Frozen protocol

- Data: {config.n_train} balanced train + {config.n_test} held-out scenes from the immutable Phase 0-B selection.
- Availability: standardized multinomial logistic regression (`C={config.probe_c}`) on the pre-answer decision token at L00/L08/L16/L24/L32.
- Behavior: after teacher-forcing the tokenizer-identical shared answer-prefix token, argmax of the frozen LM head over the four relation tokens at their actual answer position.
- Attention: decision-to-vision attention on all {config.attention_test_samples} test scenes; supporting only.
- Intervention: decision-token activation blending at L08/L16/L24. The primary causal test was fixed at L{config.primary_patch_layer}, α={config.primary_patch_alpha:.2f}, before results.
- Uncertainty: {config.bootstrap_samples} fixed-seed bootstrap resamples.
- Forbidden operations: no fine-tuning, model-structure change, adapter, or added loss.

## 3. Availability–utilization result

| State | Availability | Behavior | Gap | Availability on behavior errors |
| --- | ---: | ---: | ---: | ---: |
{rows}

Primary L32 held-out availability was {_format_percent(primary['accuracy'])} (95% CI {_format_percent(primary['accuracy_ci_low'])}–{_format_percent(primary['accuracy_ci_high'])}); behavioral utilization was {_format_percent(primary['behavior_accuracy'])}. The paired gap was {_format_percent(primary['availability_behavior_gap'])} (95% CI {_format_percent(primary['gap_ci_low'])}–{_format_percent(primary['gap_ci_high'])}). There were {primary['behavior_error_n']} behavioral errors; the L32 probe was correct on {_format_percent(primary['error_subset_accuracy'])} of them.

## 4. Attention and causal intervention

Attention is descriptive and cannot establish causal use. Per-image, per-layer values and correct/incorrect contrasts are in `results/attention_analysis.csv` and `results/statistics.json`.

The primary L16 full positive-donor patch recovered {_format_percent(patch['positive_recovery']['mean'])} of eligible baseline failures. The counterfactual control recovered {_format_percent(patch['control_recovery']['mean'])}; the paired positive-minus-control advantage was {_format_percent(patch['positive_minus_control']['mean'])} (95% CI {_format_percent(patch['positive_minus_control']['ci_low'])}–{_format_percent(patch['positive_minus_control']['ci_high'])}).

The registered secondary L24 full patch produced a larger positive-minus-control effect of {_format_percent(secondary_l24['positive_minus_control']['mean'])} (95% CI {_format_percent(secondary_l24['positive_minus_control']['ci_low'])}–{_format_percent(secondary_l24['positive_minus_control']['ci_high'])}). This is a strong hypothesis-generating late-layer signal, but L24 was not the run-before-results primary causal gate and therefore cannot replace the failed L16 test post hoc.

## 5. Registered decision gates

{gates}

# {decision['decision']}

Semantic Access Failure is **{decision['semantic_access_failure']}** in the registered experimental scope.

{route}

## 6. Interpretation limits

- A linear probe demonstrates decodability, not by itself causal use; this is why the registered causal-patch gate is mandatory.
- The matched donor patch changes a complete decision-token activation, not an isolated neuron or uniquely identified circuit.
- NF4 is an inference approximation. Any GO requires bf16/fp16 replication before a broad mechanism claim.
- Synthetic shapes isolate the relation variable but do not estimate prevalence on natural-image VLM tasks.
- The supplied task description omitted Sections 3–12. Protocol v1 was frozen before any Phase 0-C output; v2/v3 are measurement-integrity corrections documented below. The final protocol is recorded by hash in `decision.json`.

## 7. Assay-integrity correction

An initial execution compared the four relation-token logits before Vicuna's mandatory shared answer-space token. It produced a fixed `above` output (25% accuracy) and no eligible same-class successful donor. Tokenizer inspection showed that every actual candidate is encoded as the identical space token `29871` followed by its relation token. That execution was invalidated before scientific adjudication. Protocol v2 corrected only the answer position. Its audit then found one near-tie receiver whose unpatched answer flipped because behavior-with-attention used eager attention while patching used SDPA. Protocol v3 puts attention in a separate supporting forward, keeps behavior/hidden/patch on the same SDPA path, and adds exact baseline reproduction as an integrity gate. Data, thresholds, layers, bootstrap count, donor rule, and substantive GO gates are unchanged.
"""
    path.write_text(text, encoding="utf-8", newline="\n")


def write_outputs(
    config: Any,
    records: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    availability_rows: list[dict[str, Any]],
    availability_predictions: dict[int, list[str]],
    attention_rows: list[dict[str, Any]],
    patch_rows: list[dict[str, Any]],
    statistics: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        config.results_dir / "availability.csv",
        availability_rows,
        list(availability_rows[0].keys()),
    )
    record_by_row = {index: row for index, row in enumerate(records)}
    behavior_output: list[dict[str, Any]] = []
    for row in behavior_rows:
        row_index = int(row["row"])
        if record_by_row[row_index]["split"] != "test":
            continue
        enriched = dict(row)
        for layer in config.layer_indices:
            predicted = availability_predictions[layer][row_index]
            enriched[f"availability_prediction_L{layer:02d}"] = predicted
            enriched[f"availability_correct_L{layer:02d}"] = int(
                predicted == record_by_row[row_index]["relation"]
            )
        behavior_output.append(enriched)
    behavior_fields = list(behavior_output[0].keys())
    _write_csv(config.results_dir / "behavior.csv", behavior_output, behavior_fields)
    attention_fields = [
        "row",
        "image_id",
        "true_relation",
        "predicted_relation",
        "correct",
        "layer",
        "visual_attention_mass",
        "visual_attention_mass_std_heads",
        "visual_attention_entropy",
    ]
    _write_csv(config.results_dir / "attention_analysis.csv", attention_rows, attention_fields)
    patch_fields = [
        "image_id",
        "receiver_row",
        "true_relation",
        "baseline_prediction",
        "availability_prediction",
        "layer",
        "alpha",
        "patch_type",
        "donor_image_id",
        "donor_relation",
        "patched_prediction",
        "patched_correct",
        "target_logit_margin",
        "logit_left",
        "logit_right",
        "logit_above",
        "logit_below",
    ]
    _write_csv(config.results_dir / "activation_patch.csv", patch_rows, patch_fields)
    atomic_json(config.results_dir / "statistics.json", statistics)
    atomic_json(config.results_dir / "decision.json", decision)
    primary = next(row for row in availability_rows if row["layer"] == 32)
    _plot_semantic_gap(
        config.figures_dir / "semantic_gap.png",
        availability_rows,
        primary["behavior_accuracy"],
    )
    _plot_attention(config.figures_dir / "attention_difference.png", statistics["attention"])
    _plot_recovery(config.figures_dir / "recovery_curve.png", config, statistics["activation_patch"])
    _write_report(
        config.root / "Phase0C_Report.md", config, decision, availability_rows, statistics
    )
