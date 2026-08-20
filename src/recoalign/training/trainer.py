"""Multi-stage, resumable and auditable ReCoAlign trainer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor
from torch.nn import functional as F

from recoalign.models.recoalign import ReCoAlignLosses, ReCoAlignModel
from recoalign.reproducibility import (
    atomic_write_json,
    collect_environment,
    get_git_metadata,
    seed_everything,
    utc_now,
)

from .checkpoint import TrainingCheckpointManager
from .monitor import TrainingMonitor
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from .stages import STAGE_DEFINITIONS, apply_freeze_policy, parse_freeze_policy, parse_stage


@dataclass(frozen=True)
class TrainingConfig:
    experiment_id: str
    model: dict[str, Any]
    dataset: dict[str, Any]
    stage: str
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    learning_rate: float
    batch_size: int
    epochs: int
    freeze_policy: str
    loss_weights: dict[str, float]
    seed: int
    output_dir: str
    gradient_clip_norm: float | None
    validation_every: int
    save_every: int
    raw: dict[str, Any]

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> TrainingConfig:
        required = (
            "model",
            "dataset",
            "stage",
            "optimizer",
            "scheduler",
            "learning_rate",
            "batch_size",
            "freeze_policy",
            "loss_weights",
            "seed",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"training config is missing required fields: {', '.join(missing)}")
        if not isinstance(payload["model"], dict) or not isinstance(payload["dataset"], dict):
            raise ValueError("training model and dataset fields must be mappings")
        optimizer = _named_mapping(payload["optimizer"], "optimizer")
        scheduler = _named_mapping(payload["scheduler"], "scheduler")
        weights = _loss_weights(payload["loss_weights"])
        learning_rate = float(payload["learning_rate"])
        batch_size = int(payload["batch_size"])
        epochs = int(payload.get("epochs", 1))
        seed = int(payload["seed"])
        if learning_rate <= 0 or batch_size <= 0 or epochs <= 0 or seed < 0:
            raise ValueError(
                "learning_rate, batch_size, epochs and seed must be valid positive values"
            )
        stage = parse_stage(str(payload["stage"])).value
        freeze_policy = parse_freeze_policy(str(payload["freeze_policy"])).value
        validation_every = int(payload.get("validation_every", 1))
        save_every = int(payload.get("save_every", 1))
        if validation_every <= 0 or save_every <= 0:
            raise ValueError("validation_every and save_every must be positive")
        gradient_clip = payload.get("gradient_clip_norm")
        gradient_clip = None if gradient_clip is None else float(gradient_clip)
        if gradient_clip is not None and gradient_clip <= 0:
            raise ValueError("gradient_clip_norm must be positive when configured")
        return cls(
            experiment_id=str(payload.get("training_experiment", "TRAIN_UNREGISTERED")),
            model=dict(payload["model"]),
            dataset=dict(payload["dataset"]),
            stage=stage,
            optimizer=optimizer,
            scheduler=scheduler,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            freeze_policy=freeze_policy,
            loss_weights=weights,
            seed=seed,
            output_dir=str(payload.get("output_dir", "runs/training")),
            gradient_clip_norm=gradient_clip,
            validation_every=validation_every,
            save_every=save_every,
            raw=dict(payload),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("training config root must be a mapping")
        return cls.from_mapping(payload)


class ReCoAlignTrainer:
    """Train the interface in explicit stages without accepting graph inference input."""

    def __init__(
        self,
        model: ReCoAlignModel,
        config: TrainingConfig,
        *,
        run_dir: str | Path | None = None,
        device: str | torch.device = "cpu",
        capture_environment_metadata: bool = True,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.model.to(self.device)
        seed_everything(config.seed, deterministic=True)
        self.freeze_manifest = apply_freeze_policy(model, config.freeze_policy)
        self.model.losses = ReCoAlignLosses(
            semantic_weight=config.loss_weights["semantic"],
            structural_weight=config.loss_weights["structural"],
            reasoning_weight=config.loss_weights["reasoning"],
        )
        self.run_dir = Path(run_dir or _default_run_dir(config))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = TrainingMonitor(self.run_dir)
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.checkpoints: TrainingCheckpointManager | None = None
        self.global_step = 0
        self.start_epoch = 1
        self._prepare_artifacts(capture_environment_metadata)

    @classmethod
    def from_config(
        cls,
        config: TrainingConfig,
        *,
        run_dir: str | Path | None = None,
        device: str | torch.device = "cpu",
        capture_environment_metadata: bool = True,
    ) -> ReCoAlignTrainer:
        """Seed first, then initialize the method for reproducible weights."""

        seed_everything(config.seed, deterministic=True)
        model = ReCoAlignModel(config.raw)
        return cls(
            model,
            config,
            run_dir=run_dir,
            device=device,
            capture_environment_metadata=capture_environment_metadata,
        )

    def fit(
        self,
        train_batches: list[dict[str, Tensor | None]],
        validation_batches: list[dict[str, Tensor | None]],
        *,
        resume_from: str | Path | None = None,
        stop_after_epoch: int | None = None,
    ) -> dict[str, Any]:
        if not train_batches or not validation_batches:
            raise ValueError("training and validation batches must not be empty")
        self._audit_batch(train_batches[0])
        total_steps = self.config.epochs * len(train_batches)
        self.optimizer = build_optimizer(
            self.model,
            self.config.optimizer,
            learning_rate=self.config.learning_rate,
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            self.config.scheduler,
            total_steps=total_steps,
        )
        self.checkpoints = TrainingCheckpointManager(
            self.run_dir / "checkpoints",
            config=self.config.raw,
            dataset_manifest=_dataset_manifest(self.config.dataset),
        )
        if resume_from is not None:
            payload = self.checkpoints.load(
                resume_from,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                map_location=self.device,
            )
            self.start_epoch = int(payload["epoch"]) + 1
            self.global_step = int(payload["global_step"])
        end_epoch = min(self.config.epochs, stop_after_epoch or self.config.epochs)
        if self.start_epoch > end_epoch:
            raise ValueError("resume checkpoint is already beyond the requested final epoch")
        started_at = utc_now()
        latest_checkpoint: Path | None = None
        for epoch in range(self.start_epoch, end_epoch + 1):
            train_metrics = self._train_epoch(train_batches)
            self.monitor.log_training(epoch, train_metrics)
            validation_metrics: dict[str, float] = {}
            if epoch % self.config.validation_every == 0:
                validation_metrics = self.evaluate(validation_batches)
                self.monitor.log_validation(epoch, validation_metrics)
            if epoch % self.config.save_every == 0 or epoch == end_epoch:
                latest_checkpoint = self.checkpoints.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    global_step=self.global_step,
                    metrics={"training": train_metrics, "validation": validation_metrics},
                )
        summary = self._summary(started_at, end_epoch, latest_checkpoint)
        self.monitor.write_summary(summary)
        return summary

    def evaluate(self, batches: list[dict[str, Tensor | None]]) -> dict[str, float]:
        self.model.eval()
        aggregates: dict[str, list[float]] = {
            "total_loss": [],
            "reasoning_accuracy": [],
            "semantic_cosine": [],
            "object_probe_accuracy": [],
            "attribute_probe_accuracy": [],
            "relation_probe_accuracy": [],
            "composition_probe_accuracy": [],
        }
        with torch.no_grad():
            for raw_batch in batches:
                batch = self._to_device(raw_batch)
                outputs = self.model(batch["visual_tokens"])
                terms = self.model.compute_loss(
                    outputs, **_loss_arguments(batch, self.config.loss_weights)
                )
                aggregates["total_loss"].append(float(terms["total"].cpu()))
                aggregates["semantic_cosine"].append(
                    float(
                        F.cosine_similarity(
                            outputs["visual_summary"], outputs["structure_summary"], dim=-1
                        )
                        .mean()
                        .cpu()
                    )
                )
                if batch.get("answer_labels") is not None:
                    aggregates["reasoning_accuracy"].append(
                        _accuracy(outputs["reasoning_logits"], batch["answer_labels"])
                    )
                heads = outputs["structural_logits_by_type"]
                for name in ("object", "attribute", "relation", "composition"):
                    target = batch.get(f"{name}_targets")
                    if target is not None:
                        aggregates[f"{name}_probe_accuracy"].append(
                            _accuracy(heads[name], target)
                        )
        metrics = {
            name: sum(values) / len(values) for name, values in aggregates.items() if values
        }
        probe_values = [
            value for name, value in metrics.items() if name.endswith("_probe_accuracy")
        ]
        if probe_values:
            metrics["structure_probe_accuracy"] = sum(probe_values) / len(probe_values)
        return metrics

    def _train_epoch(self, batches: list[dict[str, Tensor | None]]) -> dict[str, float]:
        assert self.optimizer is not None and self.scheduler is not None
        self.model.train()
        history = {name: [] for name in self.model.loss_names}
        history["total"] = []
        for raw_batch in batches:
            batch = self._to_device(raw_batch)
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(batch["visual_tokens"])
            terms = self.model.compute_loss(
                outputs, **_loss_arguments(batch, self.config.loss_weights)
            )
            terms["total"].backward()
            if self.config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1
            for name in history:
                history[name].append(float(terms[name].detach().cpu()))
        return {
            "total_loss": _mean(history["total"]),
            "semantic_loss": _mean(history["semantic_preservation"]),
            "structural_loss": _mean(history["structural_consistency"]),
            "reasoning_loss": _mean(history["reasoning_alignment"]),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }

    def _audit_batch(self, batch: dict[str, Tensor | None]) -> None:
        forbidden = {"graph", "oracle_graph", "scene_graph", "graph_tokens"} & set(batch)
        if forbidden:
            raise ValueError(
                "oracle graph inference input is forbidden; use structural target fields only: "
                + ", ".join(sorted(forbidden))
            )
        if "visual_tokens" not in batch:
            raise ValueError("training batches require visual_tokens")
        stage = STAGE_DEFINITIONS[parse_stage(self.config.stage)]
        if (
            "semantic_targets" in stage.required_supervision
            and batch.get("semantic_targets") is None
        ):
            raise ValueError(f"{stage.stage.value} requires semantic_targets")
        if "structural_targets" in stage.required_supervision and not any(
            batch.get(f"{name}_targets") is not None
            for name in ("object", "attribute", "relation", "composition")
        ):
            raise ValueError(f"{stage.stage.value} requires structural supervision targets")
        if "answer_labels" in stage.required_supervision and batch.get("answer_labels") is None:
            raise ValueError(f"{stage.stage.value} requires answer_labels")

    def _to_device(self, batch: dict[str, Tensor | None]) -> dict[str, Tensor | None]:
        return {
            key: value.to(self.device) if isinstance(value, Tensor) else value
            for key, value in batch.items()
        }

    def _prepare_artifacts(self, capture_environment_metadata: bool) -> None:
        resolved = self.run_dir / "config.resolved.yaml"
        serialized = yaml.safe_dump(self.config.raw, sort_keys=True)
        if resolved.is_file() and resolved.read_text(encoding="utf-8") != serialized:
            raise ValueError("existing run directory has a different resolved training config")
        resolved.write_text(serialized, encoding="utf-8")
        manifest = _dataset_manifest(self.config.dataset)
        (self.run_dir / "dataset_manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8"
        )
        git = get_git_metadata()
        (self.run_dir / "git_commit.txt").write_text(
            str(git.get("commit") or "unavailable") + "\n", encoding="utf-8"
        )
        environment_path = self.run_dir / "environment.json"
        if not environment_path.is_file():
            environment = (
                collect_environment()
                if capture_environment_metadata
                else {
                    "schema_version": 1,
                    "capture_skipped": True,
                    "reason": "lightweight framework validation",
                    "git": git,
                }
            )
            atomic_write_json(environment_path, environment)
        atomic_write_json(
            self.run_dir / "training_provenance.json",
            {
                "training_experiment": self.config.experiment_id,
                "stage": self.config.stage,
                "scientific_goal": STAGE_DEFINITIONS[
                    parse_stage(self.config.stage)
                ].scientific_goal,
                "supervision_source": self.config.dataset.get("supervision"),
                "oracle_graph_used_at_inference": False,
                "extra_data_allowed": False,
                "freeze_policy": self.freeze_manifest,
                "model_fairness_keys": {
                    key: self.config.model.get(key)
                    for key in ("vision_backbone", "llm_backbone", "dataset_family")
                },
            },
        )

    def _summary(
        self,
        started_at: str,
        end_epoch: int,
        checkpoint: Path | None,
    ) -> dict[str, Any]:
        training = self.monitor.training_history
        validation = self.monitor.validation_history
        initial_loss = float(training[0]["total_loss"])
        final_loss = float(training[-1]["total_loss"])
        return {
            "schema_version": 1,
            "training_experiment": self.config.experiment_id,
            "stage": self.config.stage,
            "started_at": started_at,
            "completed_at": utc_now(),
            "epochs_completed": end_epoch,
            "global_step": self.global_step,
            "initial_total_loss": initial_loss,
            "final_total_loss": final_loss,
            "loss_decreased": final_loss < initial_loss,
            "latest_validation": validation[-1] if validation else {},
            "checkpoint": str(checkpoint) if checkpoint else None,
            "freeze_policy": self.freeze_manifest,
            "oracle_graph_used_at_inference": False,
            "scientific_efficacy": "not_established_by_training_sanity_run",
        }


def load_training_config(path: str | Path) -> TrainingConfig:
    return TrainingConfig.from_yaml(path)


def _named_mapping(value: Any, field: str) -> dict[str, Any]:
    payload = {"name": value} if isinstance(value, str) else value
    if not isinstance(payload, dict) or not str(payload.get("name", "")).strip():
        raise ValueError(f"{field} must be a name or mapping containing name")
    return dict(payload)


def _loss_weights(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("loss_weights must be a mapping")
    aliases = {
        "semantic_preservation": "semantic",
        "structural_consistency": "structural",
        "reasoning_alignment": "reasoning",
    }
    normalized = {aliases.get(key, key): float(weight) for key, weight in value.items()}
    missing = {"semantic", "structural", "reasoning"} - set(normalized)
    if missing or any(weight < 0 for weight in normalized.values()) or not any(normalized.values()):
        raise ValueError("loss_weights require non-negative semantic/structural/reasoning values")
    return {name: normalized[name] for name in ("semantic", "structural", "reasoning")}


def _loss_arguments(
    batch: dict[str, Tensor | None], weights: dict[str, float]
) -> dict[str, Tensor | None]:
    use_structural = weights["structural"] > 0
    return {
        "semantic_targets": batch.get("semantic_targets") if weights["semantic"] > 0 else None,
        "object_targets": batch.get("object_targets") if use_structural else None,
        "attribute_targets": batch.get("attribute_targets") if use_structural else None,
        "relation_targets": batch.get("relation_targets") if use_structural else None,
        "composition_targets": batch.get("composition_targets") if use_structural else None,
        "answer_labels": batch.get("answer_labels") if weights["reasoning"] > 0 else None,
    }


def _dataset_manifest(dataset: dict[str, Any]) -> dict[str, Any]:
    path = dataset.get("manifest")
    if not path:
        return {"name": dataset.get("name"), "status": "manifest_not_provided"}
    source = Path(str(path))
    if not source.is_file():
        raise FileNotFoundError(f"training dataset manifest does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training dataset manifest root must be a mapping")
    return payload


def _default_run_dir(config: TrainingConfig) -> Path:
    return Path(config.output_dir) / f"{config.experiment_id.lower()}_seed{config.seed}"


def _accuracy(logits: Tensor, targets: Tensor) -> float:
    return float((logits.argmax(dim=-1) == targets.long()).float().mean().cpu())


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


__all__ = ["ReCoAlignTrainer", "TrainingConfig", "load_training_config"]
