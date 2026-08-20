"""Controlled overfit, random-label, no-structure, and resume checks."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from recoalign.models.recoalign import ReCoAlignConfig
from recoalign.reproducibility import atomic_write_json

from .ablations import apply_ablation, randomize_structural_labels
from .recoalign_toy import make_toy_batch
from .trainer import ReCoAlignTrainer, TrainingConfig


def run_training_sanity_suite(
    config: dict[str, Any],
    *,
    output_dir: str | Path,
    capture_environment_metadata: bool = False,
) -> dict[str, Any]:
    """Execute all framework sanity checks on one controlled 100-sample split."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    base = deepcopy(config)
    base["batch_size"] = 100
    base["epochs"] = max(4, int(base.get("epochs", 4)))
    base["save_every"] = int(base["epochs"])
    training_config = TrainingConfig.from_mapping(base)
    model_config = ReCoAlignConfig.from_mapping(base)
    train_batch = make_toy_batch(
        batch_size=100,
        config=model_config,
        seed=training_config.seed + 1,
    ).as_dict()
    validation_batch = make_toy_batch(
        batch_size=100,
        config=model_config,
        seed=training_config.seed + 2,
    ).as_dict()
    clean = _run(
        base,
        root / "clean_labels",
        train_batch,
        validation_batch,
        capture_environment_metadata,
    )
    randomized = _run(
        base,
        root / "random_labels",
        randomize_structural_labels(train_batch, seed=training_config.seed + 3),
        validation_batch,
        capture_environment_metadata,
    )
    no_structure_config = apply_ablation(base, "no_structure")
    no_structure = _run(
        no_structure_config,
        root / "no_structure",
        train_batch,
        validation_batch,
        capture_environment_metadata,
    )
    resume = _resume_check(
        base,
        root / "resume",
        train_batch,
        validation_batch,
        capture_environment_metadata,
    )
    clean_probe = _probe(clean)
    random_probe = _probe(randomized)
    result = {
        "schema_version": 1,
        "samples": 100,
        "overfit": {
            "initial_loss": clean["initial_total_loss"],
            "final_loss": clean["final_total_loss"],
            "passed": clean["loss_decreased"],
        },
        "random_label": {
            "clean_structure_probe": clean_probe,
            "random_structure_probe": random_probe,
            "expected_decline_observed": clean_probe > random_probe,
        },
        "no_structure": {
            "structure_probe": _probe(no_structure),
            "delta_from_clean": clean_probe - _probe(no_structure),
            "analyzable": True,
        },
        "resume": resume,
        "oracle_graph_used_at_inference": False,
        "scientific_efficacy": "not_established_by_sanity_suite",
    }
    atomic_write_json(root / "sanity_report.json", result)
    return result


def _run(
    payload: dict[str, Any],
    run_dir: Path,
    train_batch: dict,
    validation_batch: dict,
    capture_environment_metadata: bool,
) -> dict[str, Any]:
    config = TrainingConfig.from_mapping(payload)
    trainer = ReCoAlignTrainer.from_config(
        config,
        run_dir=run_dir,
        capture_environment_metadata=capture_environment_metadata,
    )
    return trainer.fit([train_batch], [validation_batch])


def _resume_check(
    payload: dict[str, Any],
    run_dir: Path,
    train_batch: dict,
    validation_batch: dict,
    capture_environment_metadata: bool,
) -> dict[str, Any]:
    config = TrainingConfig.from_mapping(payload)
    split_epoch = max(1, config.epochs // 2)
    first = ReCoAlignTrainer.from_config(
        config,
        run_dir=run_dir,
        capture_environment_metadata=capture_environment_metadata,
    )
    partial = first.fit(
        [train_batch],
        [validation_batch],
        stop_after_epoch=split_epoch,
    )
    resumed = ReCoAlignTrainer.from_config(
        config,
        run_dir=run_dir,
        capture_environment_metadata=capture_environment_metadata,
    )
    completed = resumed.fit(
        [train_batch],
        [validation_batch],
        resume_from=partial["checkpoint"],
    )
    expected_steps = config.epochs
    return {
        "partial_epoch": split_epoch,
        "completed_epoch": completed["epochs_completed"],
        "global_step": completed["global_step"],
        "expected_global_step": expected_steps,
        "passed": completed["global_step"] == expected_steps,
    }


def _probe(result: dict[str, Any]) -> float:
    return float(result.get("latest_validation", {}).get("structure_probe_accuracy", 0.0))


__all__ = ["run_training_sanity_suite"]
