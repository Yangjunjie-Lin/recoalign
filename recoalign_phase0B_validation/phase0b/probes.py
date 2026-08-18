from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig

SEMANTICS = ("object", "attribute", "relation", "composition")


def _load_feature_stages(config: ExperimentConfig) -> dict[str, np.ndarray]:
    stages: dict[str, np.ndarray] = {
        "Za": np.load(config.features_dir / "za_mean.npy", mmap_mode="r")
    }
    for layer in config.layer_indices:
        stages[f"L{layer:02d}_visual"] = np.load(
            config.features_dir / f"visual_layer_{layer:02d}.npy", mmap_mode="r"
        )
        stages[f"L{layer:02d}_decision"] = np.load(
            config.features_dir / f"decision_layer_{layer:02d}.npy", mmap_mode="r"
        )
    return stages


def _accuracy_bootstrap(correct: np.ndarray, *, samples: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(correct)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sample = rng.integers(0, n, size=n)
        values[index] = correct[sample].mean()
    return {
        "bootstrap_mean": float(values.mean()),
        "bootstrap_std": float(values.std(ddof=1)),
        "ci95_low": float(np.percentile(values, 2.5)),
        "ci95_high": float(np.percentile(values, 97.5)),
    }


def _drop_bootstrap(
    baseline_correct: np.ndarray,
    stage_correct: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(stage_correct)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sample = rng.integers(0, n, size=n)
        values[index] = baseline_correct[sample].mean() - stage_correct[sample].mean()
    observed = float(baseline_correct.mean() - stage_correct.mean())
    return {
        "drop": observed,
        "bootstrap_mean": float(values.mean()),
        "bootstrap_std": float(values.std(ddof=1)),
        "ci95_low": float(np.percentile(values, 2.5)),
        "ci95_high": float(np.percentile(values, 97.5)),
    }


def _selectivity_bootstrap(
    relation_baseline: np.ndarray,
    relation_stage: np.ndarray,
    object_baseline: np.ndarray,
    object_stage: np.ndarray,
    *,
    samples: int,
    seed: int,
    comparison: str,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(relation_stage)
    rel_delta = relation_baseline.astype(np.int8) - relation_stage.astype(np.int8)
    obj_delta = object_baseline.astype(np.int8) - object_stage.astype(np.int8)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sample = rng.integers(0, n, size=n)
        values[index] = rel_delta[sample].mean() - obj_delta[sample].mean()
    observed = float(rel_delta.mean() - obj_delta.mean())
    return {
        "comparison": comparison,
        "observed_difference": observed,
        "bootstrap_mean": float(values.mean()),
        "bootstrap_std": float(values.std(ddof=1)),
        "ci95_low": float(np.percentile(values, 2.5)),
        "ci95_high": float(np.percentile(values, 97.5)),
        "p_one_sided": float((np.count_nonzero(values <= 0) + 1) / (samples + 1)),
    }


def _fit_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, bool]:
    probe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "linear_probe",
                LogisticRegression(
                    C=1.0,
                    max_iter=500,
                    solver="lbfgs",
                    random_state=seed,
                    tol=1e-4,
                ),
            ),
        ]
    )
    converged = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        probe.fit(np.asarray(x_train, dtype=np.float32), y_train)
        converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    return probe.predict(np.asarray(x_test, dtype=np.float32)), converged


def run_probes(config: ExperimentConfig, records: list[dict[str, Any]]) -> dict[str, Any]:
    stages = _load_feature_stages(config)
    train_mask = np.asarray([row["split"] == "train" for row in records])
    test_mask = ~train_mask
    test_records = [row for row in records if row["split"] == "test"]
    rows: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, np.ndarray]] = {}
    convergence: dict[str, bool] = {}

    for stage_index, (stage, features) in enumerate(stages.items()):
        predictions[stage] = {}
        for semantic_index, semantic in enumerate(SEMANTICS):
            labels = np.asarray([row["semantic_label"][semantic] for row in records])
            prediction, converged = _fit_probe(
                features[train_mask],
                labels[train_mask],
                features[test_mask],
                seed=config.seed + stage_index * 101 + semantic_index,
            )
            predictions[stage][semantic] = prediction
            convergence[f"{stage}:{semantic}"] = converged
            correct = prediction == labels[test_mask]
            uncertainty = _accuracy_bootstrap(
                correct,
                samples=config.bootstrap_samples,
                seed=config.seed + 10_000 + stage_index * 101 + semantic_index,
            )
            rows.append(
                {
                    "stage": stage,
                    "representation": (
                        "projected_visual"
                        if stage == "Za"
                        else "visual_hidden"
                        if stage.endswith("visual")
                        else "decision_hidden"
                    ),
                    "layer": ("Za" if stage == "Za" else int(stage[1:3])),
                    "semantic": semantic,
                    "accuracy": float(correct.mean()),
                    "accuracy_bootstrap_std": uncertainty["bootstrap_std"],
                    "accuracy_ci95_low": uncertainty["ci95_low"],
                    "accuracy_ci95_high": uncertainty["ci95_high"],
                    "converged": converged,
                }
            )

    drops: dict[str, dict[str, dict[str, float]]] = {}
    selectivity: dict[str, dict[str, dict[str, float]]] = {}
    test_labels = {
        semantic: np.asarray([row["semantic_label"][semantic] for row in test_records])
        for semantic in SEMANTICS
    }
    baseline_correct = {
        semantic: predictions["Za"][semantic] == test_labels[semantic] for semantic in SEMANTICS
    }
    for stage_index, stage in enumerate(stages):
        drops[stage] = {}
        for semantic_index, semantic in enumerate(SEMANTICS):
            correct = predictions[stage][semantic] == test_labels[semantic]
            drops[stage][semantic] = _drop_bootstrap(
                baseline_correct[semantic],
                correct,
                samples=config.bootstrap_samples,
                seed=config.seed + 20_000 + stage_index * 101 + semantic_index,
            )
        if stage.endswith("visual") and stage != "L00_visual":
            selectivity[stage] = {}
            object_stage = predictions[stage]["object"] == test_labels["object"]
            for semantic_index, semantic in enumerate(("relation", "composition")):
                semantic_stage = predictions[stage][semantic] == test_labels[semantic]
                selectivity[stage][semantic] = _selectivity_bootstrap(
                    baseline_correct[semantic],
                    semantic_stage,
                    baseline_correct["object"],
                    object_stage,
                    samples=config.bootstrap_samples,
                    seed=config.seed + 30_000 + stage_index * 101 + semantic_index,
                    comparison=f"{semantic}_drop_minus_object_drop",
                )

    prediction_dir = config.results_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for stage, semantic_predictions in predictions.items():
        path = prediction_dir / f"{stage}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row_index, record in enumerate(test_records):
                output = {
                    "image_id": record["image_id"],
                    "labels": record["semantic_label"],
                    "predictions": {
                        semantic: str(semantic_predictions[semantic][row_index])
                        for semantic in SEMANTICS
                    },
                }
                handle.write(json.dumps(output, sort_keys=True) + "\n")

    za = np.asarray(stages["Za"], dtype=np.float32)
    layer0 = np.asarray(stages["L00_visual"], dtype=np.float32)
    identity = {
        "max_abs_difference": float(np.max(np.abs(za - layer0))),
        "mean_abs_difference": float(np.mean(np.abs(za - layer0))),
        "cosine_min": float(
            np.min(
                np.sum(za * layer0, axis=1)
                / (np.linalg.norm(za, axis=1) * np.linalg.norm(layer0, axis=1) + 1e-12)
            )
        ),
    }
    return {
        "rows": rows,
        "drops": drops,
        "selectivity": selectivity,
        "identity": identity,
        "convergence": convergence,
        "all_probes_converged": all(convergence.values()),
    }
