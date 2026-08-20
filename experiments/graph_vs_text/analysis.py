"""EXP001 metrics: paired controls, depth slices, errors, and token efficiency."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any

from evaluation.metrics import paired_difference

PRIMARY_CONDITIONS = ("image_only", "object_list", "caption", "scene_graph")
SETTINGS = ("natural", "token_matched")


def analyze_seed(
    rows: list[dict[str, Any]],
    ood_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 2000,
    statistics_seed: int = 20260819,
) -> dict[str, Any]:
    """Compute all preregistered seed-level metrics from prediction rows."""

    primary = [row for row in rows if row.get("scope") == "iid_primary"]
    natural = [row for row in primary if row["setting"] == "natural"]
    metrics: dict[str, Any] = {
        "experiment": "graph_vs_text",
        "conditions": _condition_table(natural),
        "contrasts": {
            "contrasts": {
                "graph_over_text": _contrast(
                    natural,
                    "scene_graph",
                    "caption",
                    bootstrap_samples=bootstrap_samples,
                    seed=statistics_seed,
                ),
            }
        },
        "controlled_contrasts": {},
        "reasoning_depth": {},
        "task_accuracy": {},
        "error_taxonomy": _error_taxonomy(primary),
        "token_accounting": _token_accounting(primary),
        "information_control": _information_control(primary),
        "ablations": _ablation_metrics(rows),
        "ood_contrasts": {},
    }
    for setting in SETTINGS:
        selected = [row for row in primary if row["setting"] == setting]
        metrics["controlled_contrasts"][setting] = {
            "graph_over_caption": _contrast(
                selected,
                "scene_graph",
                "caption",
                bootstrap_samples=bootstrap_samples,
                seed=statistics_seed + len(setting),
            )
        }
        metrics["task_accuracy"][setting] = _task_accuracy(selected)
        metrics["reasoning_depth"][setting] = _depth_accuracy(
            selected,
            bootstrap_samples=bootstrap_samples,
            seed=statistics_seed + 101 * len(setting),
        )
        selected_ood = [row for row in ood_rows if row["setting"] == setting]
        metrics["ood_contrasts"][setting] = {
            "graph_over_caption": _contrast(
                selected_ood,
                "scene_graph",
                "caption",
                bootstrap_samples=bootstrap_samples,
                seed=statistics_seed + 211 * len(setting),
            )
        }
    metrics["dataset_size"] = len({row["scene_id"] for row in primary})
    metrics["prediction_count"] = len(rows) + len(ood_rows)
    return metrics


def _condition_table(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)
    return {
        condition: {
            "accuracy": _accuracy(values),
            "n": len(values),
        }
        for condition, values in sorted(grouped.items())
    }


def _contrast(
    rows: list[dict[str, Any]],
    first: str,
    second: str,
    *,
    bootstrap_samples: int = 2000,
    seed: int = 20260819,
) -> dict[str, Any]:
    selected = [row for row in rows if row["condition"] in {first, second}]
    return paired_difference(
        selected,
        first,
        second,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


def _depth_accuracy(
    rows: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for depth in range(1, 5):
        selected = [row for row in rows if int(row["hop_depth"]) == depth]
        result[str(depth)] = {
            "conditions": _condition_table(selected),
            "graph_over_caption": _contrast(
                selected,
                "scene_graph",
                "caption",
                bootstrap_samples=bootstrap_samples,
                seed=seed + depth,
            ),
        }
    return result


def _task_accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_type"])].append(row)
    return {
        task: {"conditions": _condition_table(values), "n": len(values)}
        for task, values in sorted(grouped.items())
    }


def _error_taxonomy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mapping = {
        "object_reasoning": "object_error",
        "attribute_reasoning": "object_error",
        "relation_reasoning": "relation_error",
        "multi_hop": "compositional_error",
    }
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    for row in rows:
        key = f"{row['setting']}:{row['condition']}"
        totals[key] += 1
        if not row["correct"]:
            counts[key][mapping[str(row["question_type"])]] += 1
    return {
        key: {
            "n": totals[key],
            "counts": {
                label: counts[key][label]
                for label in ("object_error", "relation_error", "compositional_error")
            },
            "error_rate": sum(counts[key].values()) / totals[key],
        }
        for key in sorted(totals)
    }


def _token_accounting(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['setting']}:{row['condition']}"].append(row)
    result: dict[str, Any] = {}
    for key, values in sorted(grouped.items()):
        mean_input = statistics.fmean(float(row["input"]["input_tokens"]) for row in values)
        accuracy = _accuracy(values)
        result[key] = {
            "mean_input_tokens": mean_input,
            "mean_evidence_tokens": statistics.fmean(
                float(row["input"]["evidence_tokens"]) for row in values
            ),
            "mean_semantic_units": statistics.fmean(
                float(row["input"]["semantic_units"]) for row in values
            ),
            "mean_relation_count": statistics.fmean(
                float(row["input"]["relation_count"]) for row in values
            ),
            "accuracy": accuracy,
            "accuracy_per_1000_input_tokens": (
                accuracy * 1000.0 / mean_input if mean_input else 0.0
            ),
            "n": len(values),
        }
    return result


def _information_control(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["condition"] in {"caption", "scene_graph"}:
            grouped[(str(row["scene_id"]), str(row["setting"]))][str(row["condition"])] = row
    mismatched_hashes = 0
    token_deltas: list[int] = []
    incomplete_pairs = 0
    for (_scene_id, setting), pair in grouped.items():
        if set(pair) != {"caption", "scene_graph"}:
            incomplete_pairs += 1
            continue
        if (
            pair["caption"]["input"]["semantic_facts_sha256"]
            != pair["scene_graph"]["input"]["semantic_facts_sha256"]
        ):
            mismatched_hashes += 1
        if setting == "token_matched":
            token_deltas.append(
                abs(
                    int(pair["caption"]["input"]["input_tokens"])
                    - int(pair["scene_graph"]["input"]["input_tokens"])
                )
            )
    return {
        "pair_count": len(grouped),
        "incomplete_pair_count": incomplete_pairs,
        "semantic_hash_mismatch_count": mismatched_hashes,
        "token_matched_pair_count": len(token_deltas),
        "maximum_token_delta": max(token_deltas, default=0),
        "mean_token_delta": statistics.fmean(token_deltas) if token_deltas else 0.0,
        "passed": incomplete_pairs == 0 and mismatched_hashes == 0,
    }


def _ablation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ablations = [row for row in rows if row.get("scope") == "ablation"]
    canonical = [
        row
        for row in rows
        if row.get("scope") == "iid_primary"
        and row["setting"] == "natural"
        and row["condition"] == "scene_graph"
    ]
    all_rows = [*canonical, *ablations]
    variants = (
        "graph_order_shuffled",
        "text_unordered",
        "graph_serialization_triples",
        "graph_serialization_json",
    )
    return {
        "conditions": _condition_table(all_rows),
        "contrasts": {
            f"scene_graph_over_{variant}": _contrast(all_rows, "scene_graph", variant)
            for variant in variants
        },
    }


def _accuracy(rows: list[dict[str, Any]]) -> float:
    return sum(bool(row["correct"]) for row in rows) / len(rows) if rows else 0.0


__all__ = ["PRIMARY_CONDITIONS", "SETTINGS", "analyze_seed"]
