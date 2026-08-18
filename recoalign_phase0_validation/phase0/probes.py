from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig
from .data import read_jsonl
from .features import load_features
from .statistics import bootstrap_accuracy, bootstrap_drop, bootstrap_drop_comparison

SEMANTICS = ("object", "attribute", "relation", "composition")
STAGES = ("Zv", "Za")


@dataclass
class TrainedProbe:
    scaler: StandardScaler
    classifier: LogisticRegression
    classes: list[str]

    def predict_labels(self, features: np.ndarray) -> np.ndarray:
        transformed = self.scaler.transform(np.asarray(features, dtype=np.float32))
        indices = self.classifier.predict(transformed)
        return np.asarray([self.classes[int(index)] for index in indices], dtype=object)


def _label(record: dict[str, Any], semantic: str) -> str:
    if semantic == "object":
        return str(record["object_label"])
    if semantic == "attribute":
        return str(record["attribute_label"])
    return str(record[semantic])


def _encode_labels(
    train_labels: list[str], test_labels: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    classes = sorted(set(train_labels))
    mapping = {label: index for index, label in enumerate(classes)}
    unseen = sorted(set(test_labels) - set(classes))
    if unseen:
        raise RuntimeError(f"Test contains unseen labels: {unseen[:3]}")
    return (
        np.asarray([mapping[label] for label in train_labels], dtype=np.int64),
        np.asarray([mapping[label] for label in test_labels], dtype=np.int64),
        classes,
    )


def _fit_one(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: list[str],
    test_labels: list[str],
    *,
    seed: int,
    max_iter: int,
) -> tuple[TrainedProbe, np.ndarray, np.ndarray, bool, int]:
    y_train, y_test, classes = _encode_labels(train_labels, test_labels)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(np.asarray(train_features, dtype=np.float32))
    x_test = scaler.transform(np.asarray(test_features, dtype=np.float32))
    classifier = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=seed,
        tol=1e-4,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(x_train, y_train)
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    prediction_indices = classifier.predict(x_test)
    predictions = np.asarray([classes[int(index)] for index in prediction_indices], dtype=object)
    truth = np.asarray(test_labels, dtype=object)
    return (
        TrainedProbe(scaler, classifier, classes),
        predictions,
        truth,
        converged,
        int(np.max(classifier.n_iter_)),
    )


def _save_predictions(
    output_dir: Path,
    *,
    image_ids: list[str],
    truth: np.ndarray,
    predictions: np.ndarray,
    stage: str,
    semantic: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{stage.lower()}_{semantic}.npz",
        image_id=np.asarray(image_ids, dtype=str),
        truth=np.asarray(truth, dtype=str),
        prediction=np.asarray(predictions, dtype=str),
    )


def run_probes(config: ExperimentConfig) -> dict[str, Any]:
    records = read_jsonl(config.dataset_dir / "metadata.jsonl")
    train_indices = np.asarray(
        [i for i, row in enumerate(records) if row["split"] == "train"], dtype=np.int64
    )
    test_indices = np.asarray(
        [i for i, row in enumerate(records) if row["split"] == "test"], dtype=np.int64
    )
    train_records = [records[int(i)] for i in train_indices]
    test_records = [records[int(i)] for i in test_indices]
    test_image_ids = [str(row["image_id"]) for row in test_records]

    rows: list[dict[str, Any]] = []
    correct: dict[str, dict[str, np.ndarray]] = {semantic: {} for semantic in SEMANTICS}
    probes: dict[str, dict[str, TrainedProbe]] = {stage: {} for stage in STAGES}

    for stage_index, stage in enumerate(STAGES):
        features = load_features(config, "main", stage)
        train_features = features[train_indices]
        test_features = features[test_indices]
        for semantic_index, semantic in enumerate(SEMANTICS):
            train_labels = [_label(row, semantic) for row in train_records]
            test_labels = [_label(row, semantic) for row in test_records]
            probe, predictions, truth, converged, iterations = _fit_one(
                train_features,
                test_features,
                train_labels,
                test_labels,
                seed=config.seed + stage_index * 100 + semantic_index,
                max_iter=config.probe_max_iter,
            )
            probes[stage][semantic] = probe
            is_correct = predictions == truth
            correct[semantic][stage] = is_correct
            stats = bootstrap_accuracy(
                is_correct,
                samples=config.bootstrap_samples,
                seed=config.seed + 1_000 + stage_index * 100 + semantic_index,
            )
            rows.append(
                {
                    "evaluation": "main",
                    "semantic": semantic,
                    "stage": stage,
                    "accuracy": float(accuracy_score(truth, predictions)),
                    **stats,
                    "drop_from_zv": None,
                    "n_train": len(train_records),
                    "n_test": len(test_records),
                    "num_classes": len(probe.classes),
                    "pair_accuracy": None,
                    "converged": converged,
                    "iterations": iterations,
                }
            )
            _save_predictions(
                config.results_dir / "predictions",
                image_ids=test_image_ids,
                truth=truth,
                predictions=predictions,
                stage=stage,
                semantic=semantic,
            )

    drops: dict[str, dict[str, float]] = {}
    for semantic_index, semantic in enumerate(SEMANTICS):
        drop_stats = bootstrap_drop(
            correct[semantic]["Zv"],
            correct[semantic]["Za"],
            samples=config.bootstrap_samples,
            seed=config.seed + 2_000 + semantic_index,
        )
        drops[semantic] = drop_stats
        for row in rows:
            if row["evaluation"] == "main" and row["semantic"] == semantic and row["stage"] == "Za":
                row["drop_from_zv"] = drop_stats["drop"]

    comparison = bootstrap_drop_comparison(
        object_zv=correct["object"]["Zv"],
        object_za=correct["object"]["Za"],
        relation_zv=correct["relation"]["Zv"],
        relation_za=correct["relation"]["Za"],
        samples=config.bootstrap_samples,
        seed=config.seed + 3_000,
    )

    control_records = read_jsonl(config.dataset_dir / "control_metadata.jsonl")
    control_truth = np.asarray([_label(row, "relation") for row in control_records], dtype=object)
    control_image_ids = [str(row["image_id"]) for row in control_records]
    control_correct: dict[str, np.ndarray] = {}
    for stage_index, stage in enumerate(STAGES):
        features = load_features(config, "control", stage)
        predictions = probes[stage]["relation"].predict_labels(features)
        is_correct = predictions == control_truth
        control_correct[stage] = is_correct
        pair_correct = np.asarray(
            [
                bool(is_correct[index] and is_correct[index + 1])
                for index in range(0, len(is_correct), 2)
            ],
            dtype=np.float64,
        )
        stats = bootstrap_accuracy(
            is_correct,
            samples=config.bootstrap_samples,
            seed=config.seed + 4_000 + stage_index,
        )
        rows.append(
            {
                "evaluation": "semantic_isolation",
                "semantic": "relation",
                "stage": stage,
                "accuracy": float(np.mean(is_correct)),
                **stats,
                "drop_from_zv": None,
                "n_train": len(train_records),
                "n_test": len(control_records),
                "num_classes": len(probes[stage]["relation"].classes),
                "pair_accuracy": float(np.mean(pair_correct)),
                "converged": True,
                "iterations": None,
            }
        )
        _save_predictions(
            config.results_dir / "predictions",
            image_ids=control_image_ids,
            truth=control_truth,
            predictions=predictions,
            stage=stage,
            semantic="relation_isolation",
        )

    isolation_drop = bootstrap_drop(
        control_correct["Zv"],
        control_correct["Za"],
        samples=config.bootstrap_samples,
        seed=config.seed + 5_000,
    )
    for row in rows:
        if row["evaluation"] == "semantic_isolation" and row["stage"] == "Za":
            row["drop_from_zv"] = isolation_drop["drop"]

    payload = {
        "rows": rows,
        "drops": drops,
        "object_vs_relation": comparison,
        "semantic_isolation_drop": isolation_drop,
    }
    config.results_dir.mkdir(parents=True, exist_ok=True)
    (config.results_dir / "statistics.json").write_text(
        json.dumps(
            {
                "drops": drops,
                "object_vs_relation": comparison,
                "semantic_isolation_drop": isolation_drop,
                "bootstrap_samples": config.bootstrap_samples,
                "seed": config.seed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload
