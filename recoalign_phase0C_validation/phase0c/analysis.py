from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _mean_ci(values: np.ndarray, *, samples: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    boot = values[indices].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
    }


def run_availability_probes(
    config: Any,
    records: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, list[str]], dict[str, Any]]:
    train_indices = np.array(
        [index for index, row in enumerate(records) if row["split"] == "train"], dtype=int
    )
    test_indices = np.array(
        [index for index, row in enumerate(records) if row["split"] == "test"], dtype=int
    )
    labels = np.array([row["relation"] for row in records], dtype=object)
    behavior_by_row = {int(row["row"]): row for row in behavior_rows}
    behavior_correct = np.array(
        [int(behavior_by_row[index]["correct"]) for index in test_indices], dtype=float
    )
    predictions: dict[int, list[str]] = {}
    availability_rows: list[dict[str, Any]] = []
    per_layer: dict[str, Any] = {}

    for layer in config.layer_indices:
        features = np.asarray(
            np.load(config.features_dir / f"decision_layer_{layer:02d}.npy", mmap_mode="r"),
            dtype=np.float32,
        )
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "probe",
                    LogisticRegression(
                        C=config.probe_c,
                        max_iter=config.probe_max_iter,
                        solver="lbfgs",
                        random_state=config.seed,
                    ),
                ),
            ]
        )
        caught: list[warnings.WarningMessage]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(features[train_indices], labels[train_indices])
        predicted = model.predict(features).tolist()
        predictions[layer] = predicted
        test_probe_correct = np.array(
            [int(predicted[index] == labels[index]) for index in test_indices], dtype=float
        )
        accuracy = _mean_ci(
            test_probe_correct,
            samples=config.bootstrap_samples,
            seed=config.seed + layer * 101 + 1,
        )
        gap = _mean_ci(
            test_probe_correct - behavior_correct,
            samples=config.bootstrap_samples,
            seed=config.seed + layer * 101 + 2,
        )
        error_mask = behavior_correct == 0
        error_accuracy = _mean_ci(
            test_probe_correct[error_mask],
            samples=config.bootstrap_samples,
            seed=config.seed + layer * 101 + 3,
        )
        convergence_warnings = [
            str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)
        ]
        probe = model.named_steps["probe"]
        converged = not convergence_warnings and bool(np.all(probe.n_iter_ < config.probe_max_iter))
        row = {
            "layer": layer,
            "n_train": len(train_indices),
            "n_test": len(test_indices),
            "chance": 1.0 / len(config.choice_words),
            "accuracy": accuracy["mean"],
            "accuracy_ci_low": accuracy["ci_low"],
            "accuracy_ci_high": accuracy["ci_high"],
            "behavior_accuracy": float(behavior_correct.mean()),
            "availability_behavior_gap": gap["mean"],
            "gap_ci_low": gap["ci_low"],
            "gap_ci_high": gap["ci_high"],
            "behavior_error_n": int(error_mask.sum()),
            "error_subset_accuracy": error_accuracy["mean"],
            "error_subset_ci_low": error_accuracy["ci_low"],
            "error_subset_ci_high": error_accuracy["ci_high"],
            "probe_converged": converged,
            "probe_iterations_max": int(np.max(probe.n_iter_)),
        }
        availability_rows.append(row)
        per_layer[str(layer)] = {
            "accuracy": accuracy,
            "paired_availability_minus_behavior": gap,
            "availability_on_behavior_errors": error_accuracy,
            "probe_converged": converged,
            "convergence_warnings": convergence_warnings,
            "iterations": probe.n_iter_.tolist(),
        }

    return availability_rows, predictions, {
        "definition": (
            "standardized C=1 multinomial logistic-regression probe trained on 720 and "
            "evaluated on the fixed 288 held-out decision-position states"
        ),
        "behavior_accuracy": _mean_ci(
            behavior_correct,
            samples=config.bootstrap_samples,
            seed=config.seed + 9001,
        ),
        "per_layer": per_layer,
    }


def summarize_attention(
    config: Any, attention_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in attention_rows:
        by_layer[int(row["layer"])].append(row)
    summary: dict[str, Any] = {}
    for layer, rows in sorted(by_layer.items()):
        correct = np.array(
            [row["visual_attention_mass"] for row in rows if row["correct"]], dtype=float
        )
        incorrect = np.array(
            [row["visual_attention_mass"] for row in rows if not row["correct"]], dtype=float
        )
        correct_ci = _mean_ci(
            correct, samples=config.bootstrap_samples, seed=config.seed + 10000 + layer
        )
        incorrect_ci = _mean_ci(
            incorrect, samples=config.bootstrap_samples, seed=config.seed + 11000 + layer
        )
        if len(correct) and len(incorrect):
            rng = np.random.default_rng(config.seed + 12000 + layer)
            correct_boot = correct[
                rng.integers(0, len(correct), size=(config.bootstrap_samples, len(correct)))
            ].mean(axis=1)
            incorrect_boot = incorrect[
                rng.integers(0, len(incorrect), size=(config.bootstrap_samples, len(incorrect)))
            ].mean(axis=1)
            diff_boot = incorrect_boot - correct_boot
            difference = {
                "mean": float(incorrect.mean() - correct.mean()),
                "ci_low": float(np.quantile(diff_boot, 0.025)),
                "ci_high": float(np.quantile(diff_boot, 0.975)),
            }
        else:
            difference = {"mean": None, "ci_low": None, "ci_high": None}
        summary[str(layer)] = {
            "correct": correct_ci,
            "incorrect": incorrect_ci,
            "incorrect_minus_correct": difference,
        }
    return {
        "role": "supporting diagnostic only; excluded from GO/NO-GO",
        "n_rows": len(attention_rows),
        "n_test_images": len({row["image_id"] for row in attention_rows}),
        "per_layer": summary,
    }


def summarize_patching(config: Any, patch_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not patch_rows:
        return {
            "selected_trials": 0,
            "primary": {
                "positive_recovery": {"n": 0, "mean": None, "ci_low": None, "ci_high": None},
                "control_recovery": {"n": 0, "mean": None, "ci_low": None, "ci_high": None},
                "positive_minus_control": {
                    "n": 0,
                    "mean": None,
                    "ci_low": None,
                    "ci_high": None,
                },
            },
            "curve": {},
        }
    groups: dict[tuple[int, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in patch_rows:
        groups[(int(row["layer"]), float(row["alpha"]), row["patch_type"])].append(row)
    curve: dict[str, Any] = {}
    for (layer, alpha, patch_type), rows in sorted(groups.items()):
        values = np.array([row["patched_correct"] for row in rows], dtype=float)
        curve[f"L{layer:02d}|{alpha:.2f}|{patch_type}"] = _mean_ci(
            values,
            samples=config.bootstrap_samples,
            seed=config.seed + layer * 1000 + int(alpha * 100) * 10 + len(patch_type),
        )

    per_layer_full_patch: dict[str, Any] = {}
    for layer_index in config.patch_layers:
        layer_positive = {
            row["image_id"]: row
            for row in groups.get((layer_index, 1.0, "positive"), [])
        }
        layer_control = {
            row["image_id"]: row
            for row in groups.get((layer_index, 1.0, "control"), [])
        }
        layer_baseline = {
            row["image_id"]: row
            for row in groups.get((layer_index, 0.0, "baseline"), [])
        }
        layer_common = sorted(
            set(layer_positive) & set(layer_control) & set(layer_baseline)
        )
        positive_values = np.array(
            [layer_positive[key]["patched_correct"] for key in layer_common], dtype=float
        )
        control_values = np.array(
            [layer_control[key]["patched_correct"] for key in layer_common], dtype=float
        )
        baseline_values = np.array(
            [layer_baseline[key]["patched_correct"] for key in layer_common], dtype=float
        )
        per_layer_full_patch[str(layer_index)] = {
            "positive_incremental_recovery": _mean_ci(
                positive_values - baseline_values,
                samples=config.bootstrap_samples,
                seed=config.seed + 30000 + layer_index * 3,
            ),
            "control_incremental_recovery": _mean_ci(
                control_values - baseline_values,
                samples=config.bootstrap_samples,
                seed=config.seed + 30001 + layer_index * 3,
            ),
            "positive_minus_control": _mean_ci(
                positive_values - control_values,
                samples=config.bootstrap_samples,
                seed=config.seed + 30002 + layer_index * 3,
            ),
        }

    layer = config.primary_patch_layer
    alpha = config.primary_patch_alpha
    positive_rows = {
        row["image_id"]: row for row in groups.get((layer, alpha, "positive"), [])
    }
    control_rows = {
        row["image_id"]: row for row in groups.get((layer, alpha, "control"), [])
    }
    baseline_rows = {
        row["image_id"]: row for row in groups.get((layer, 0.0, "baseline"), [])
    }
    common = sorted(set(positive_rows) & set(control_rows) & set(baseline_rows))
    positive = np.array([positive_rows[key]["patched_correct"] for key in common], dtype=float)
    control = np.array([control_rows[key]["patched_correct"] for key in common], dtype=float)
    baseline = np.array([baseline_rows[key]["patched_correct"] for key in common], dtype=float)
    return {
        "selected_trials": len(common),
        "primary_layer": layer,
        "primary_alpha": alpha,
        "primary": {
            "positive_recovery": _mean_ci(
                positive - baseline,
                samples=config.bootstrap_samples,
                seed=config.seed + 20001,
            ),
            "control_recovery": _mean_ci(
                control - baseline,
                samples=config.bootstrap_samples,
                seed=config.seed + 20002,
            ),
            "positive_minus_control": _mean_ci(
                positive - control,
                samples=config.bootstrap_samples,
                seed=config.seed + 20003,
            ),
        },
        "curve": curve,
        "per_layer_full_patch": per_layer_full_patch,
        "primary_baseline_accuracy": _mean_ci(
            baseline, samples=config.bootstrap_samples, seed=config.seed + 20004
        ),
    }


def make_decision(
    config: Any,
    availability_rows: list[dict[str, Any]],
    patch_statistics: dict[str, Any],
    integrity: dict[str, Any],
) -> dict[str, Any]:
    primary = next(
        row for row in availability_rows if row["layer"] == config.primary_availability_layer
    )
    patch = patch_statistics["primary"]
    positive = patch["positive_recovery"]
    advantage = patch["positive_minus_control"]
    gates = {
        "full_registered_protocol": bool(
            config.scientific_protocol_valid()
            and integrity.get("behavior_complete")
            and integrity.get("attention_complete")
            and integrity.get("patch_complete")
            and integrity.get("patch_baseline_reproduced")
            and all(row["probe_converged"] for row in availability_rows)
        ),
        "representation_available": bool(
            primary["accuracy"] >= config.min_availability_accuracy
            and primary["accuracy_ci_low"] > 1.0 / len(config.choice_words)
        ),
        "availability_behavior_gap": bool(
            primary["availability_behavior_gap"] >= config.min_gap
            and primary["gap_ci_low"] > 0
        ),
        "representation_available_on_failures": bool(
            primary["behavior_error_n"] >= config.min_error_trials
            and primary["error_subset_accuracy"] is not None
            and primary["error_subset_accuracy"] >= config.min_error_availability_accuracy
            and primary["error_subset_ci_low"] > 1.0 / len(config.choice_words)
        ),
        "causal_patch_recovery": bool(
            patch_statistics["selected_trials"] >= config.min_error_trials
            and positive["mean"] is not None
            and positive["mean"] >= config.min_patch_recovery
            and positive["ci_low"] > 0
            and advantage["mean"] is not None
            and advantage["mean"] >= config.min_patch_advantage
            and advantage["ci_low"] > 0
        ),
    }
    go = all(gates.values())
    return {
        "decision": "GO" if go else "NO-GO",
        "semantic_access_failure": "ESTABLISHED" if go else "NOT ESTABLISHED",
        "scope": (
            "frozen LLaVA-1.5-7B (NF4 inference) on the fixed controlled synthetic "
            "four-way spatial-relation task"
        ),
        "gates": gates,
        "thresholds": {
            "min_availability_accuracy": config.min_availability_accuracy,
            "min_gap": config.min_gap,
            "min_error_trials": config.min_error_trials,
            "min_error_availability_accuracy": config.min_error_availability_accuracy,
            "min_patch_recovery": config.min_patch_recovery,
            "min_patch_advantage": config.min_patch_advantage,
        },
        "observed": {
            "behavior_accuracy": primary["behavior_accuracy"],
            "availability_accuracy_L32": primary["accuracy"],
            "availability_behavior_gap_L32": primary["availability_behavior_gap"],
            "behavior_error_n": primary["behavior_error_n"],
            "availability_on_errors_L32": primary["error_subset_accuracy"],
            "patch_selected_trials": patch_statistics["selected_trials"],
            "positive_patch_recovery_L16": positive["mean"],
            "patch_advantage_L16": advantage["mean"],
        },
        "route": (
            "Proceed to higher-precision, natural-image Causal Intervention replication."
            if go
            else "Terminate the ReCoAlign mechanism-discovery route."
        ),
    }
