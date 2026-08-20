"""Shared configuration, backend selection, logging, and output lifecycle."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from datasets.records import SceneRecord
from diagnosis.interface_gap_analysis.gap import summarize_interface_gap
from evaluation.metrics import evaluate_rows, write_metrics
from recoalign.models.vlm.base import BaseVLM
from recoalign.models.vlm.registry import create_vlm, validate_inline_model_config
from synthetic_world.compositional_tasks.tasks import CONDITIONS
from synthetic_world.generator.generator import GeneratorConfig, SyntheticWorldGenerator


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be a mapping: {source}")
    experiment = payload.get("experiment")
    model = payload.get("model")
    synthetic = payload.get("synthetic")
    if (
        not isinstance(experiment, dict)
        or not isinstance(model, dict)
        or not isinstance(synthetic, dict)
    ):
        raise ValueError("configuration requires experiment, model, and synthetic mappings")
    for field in ("name", "seed", "output_dir"):
        if field not in experiment:
            raise ValueError(f"configuration experiment.{field} is required")
    if not isinstance(experiment["seed"], int) or experiment["seed"] < 0:
        raise ValueError("configuration experiment.seed must be a non-negative integer")
    validate_inline_model_config(model)
    if "id" in experiment:
        for field in ("hypothesis_id", "protocol", "seeds"):
            if field not in experiment:
                raise ValueError(f"governed configuration experiment.{field} is required")
        if not isinstance(experiment["seeds"], list) or not experiment["seeds"]:
            raise ValueError("configuration experiment.seeds must be a non-empty list")
        data = payload.get("data")
        evaluation = payload.get("evaluation")
        if not isinstance(data, dict):
            raise ValueError("governed configuration requires a data mapping")
        for field in ("dataset", "version", "manifest", "split"):
            if not data.get(field):
                raise ValueError(f"governed configuration data.{field} is required")
        if not isinstance(evaluation, dict):
            raise ValueError("governed configuration requires an evaluation mapping")
        for field in (
            "conditions",
            "confidence_level",
            "bootstrap_samples",
            "significance_test",
            "alternative",
            "statistics_seed",
        ):
            if field not in evaluation:
                raise ValueError(f"governed configuration evaluation.{field} is required")
        if model.get("backend") == "reference":
            accuracy = model.get("condition_accuracy")
            conditions = evaluation["conditions"]
            if not isinstance(accuracy, dict) or any(key not in accuracy for key in conditions):
                raise ValueError(
                    "governed reference configuration requires condition_accuracy "
                    "for all conditions"
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 1.0
                for value in accuracy.values()
            ):
                raise ValueError("model.condition_accuracy values must be numeric in [0, 1]")
    if payload.get("training", {}).get("enabled") not in {None, False}:
        raise ValueError("structured experiments cannot enable training")
    payload.setdefault("meta", {})["config_path"] = str(source.resolve().as_posix())
    return payload


def seed_everything(seed: int) -> None:
    random.seed(seed)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def checkpoint_provenance(config: dict[str, Any]) -> dict[str, Any]:
    model = dict(config.get("model", {}))
    manifest_path = model.get("checkpoint_manifest")
    if not manifest_path:
        return {"status": "unbound", "backend": model.get("backend", "unknown")}
    path = Path(manifest_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if not path.exists():
        raise FileNotFoundError(f"checkpoint manifest does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint manifest must be a mapping: {path}")
    raw = path.read_bytes()
    model_path_value = model.get("model_path", model.get("model_dir"))
    model_path = _resolve_project_path(model_path_value) if model_path_value else None
    files = _checkpoint_file_verification(model_path, payload.get("files", []))
    prompt_path_value = model.get("prompt_protocol")
    prompt_path = _resolve_project_path(prompt_path_value) if prompt_path_value else None
    return {
        "status": "declared",
        "manifest": str(path.as_posix()),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "identifier": payload.get("identifier"),
        "framework": payload.get("framework"),
        "model": payload.get("model"),
        "source": payload.get("source"),
        "revision": model.get("revision", model.get("model_revision")),
        "quantization": model.get("quantization"),
        "dtype": model.get("dtype"),
        "device_map": model.get("device_map"),
        "model_config": model.get("model_config"),
        "model_config_sha256": model.get("model_config_sha256"),
        "model_path": model_path.as_posix() if model_path else None,
        "checkpoint_files": files,
        "checkpoint_fingerprint": _checkpoint_fingerprint(model_path, files),
        "generation": dict(model.get("generation", {})),
        "prompt_protocol": prompt_path.as_posix() if prompt_path else None,
        "prompt_protocol_sha256": (
            hashlib.sha256(prompt_path.read_bytes()).hexdigest()
            if prompt_path is not None and prompt_path.is_file()
            else None
        ),
    }


def build_generator(config: dict[str, Any]) -> SyntheticWorldGenerator:
    synthetic = config.get("synthetic", {})
    return SyntheticWorldGenerator(
        GeneratorConfig(
            seed=int(config["experiment"]["seed"]),
            image_size=int(synthetic.get("image_size", 128)),
            write_images=bool(synthetic.get("write_images", True)),
            style=str(synthetic.get("style", "flat")),
            min_hops=int(synthetic.get("min_hops", 1)),
            max_hops=int(synthetic.get("max_hops", 4)),
        )
    )


def build_vlm(config: dict[str, Any]) -> BaseVLM:
    model = dict(config.get("model", {}))
    return create_vlm(model, seed=int(model.get("seed", config["experiment"]["seed"])))


def run_conditions(
    records: Iterable[SceneRecord],
    vlm: BaseVLM,
    conditions: Iterable[str],
    *,
    setting: str = "natural",
    token_match_tolerance: int = 1,
) -> list[dict[str, Any]]:
    requests: list[tuple[SceneRecord, str, Any]] = []
    for record in records:
        for condition in conditions:
            prepared = vlm.prepare_input(
                record,
                condition,
                setting=setting,
                token_match_tolerance=token_match_tolerance,
            )
            requests.append((record, str(condition), prepared))
    predictions = vlm.reason_batch(requests)
    rows: list[dict[str, Any]] = []
    for (record, condition, prepared), prediction in zip(
        requests, predictions, strict=True
    ):
        input_payload = prepared.to_dict()
        input_payload.update(
            {
                "prompt": prepared.prompt,
                "prompt_sha256": hashlib.sha256(prepared.prompt.encode("utf-8")).hexdigest(),
                "image": str(prepared.image) if prepared.image is not None else None,
            }
        )
        evaluation = vlm.evaluate_detailed(prediction, record)
        rows.append(
            {
                "sample_id": record.scene_id,
                "scene_id": record.scene_id,
                "condition": condition,
                "setting": setting,
                "answer": record.answer,
                "ground_truth": record.answer,
                "prediction": prediction,
                "correct": evaluation.correct,
                "evaluation": evaluation.to_dict(),
                "seed": record.metadata.get("seed"),
                "split": record.metadata.get("split"),
                "composition": record.metadata.get("composition"),
                "question_type": record.metadata.get("question_type"),
                "hop_depth": record.metadata.get("hop_depth"),
                "input": input_payload,
            }
        )
    return rows


def runtime_provenance() -> dict[str, Any]:
    """Capture software and accelerator identity without importing model weights."""

    payload: dict[str, Any] = {
        "python": sys.version,
        "platform": sys.platform,
        "torch": None,
        "cuda": None,
        "devices": [],
    }
    try:
        import torch

        payload["torch"] = torch.__version__
        payload["cuda"] = {
            "runtime": torch.version.cuda,
            "available": torch.cuda.is_available(),
            "cudnn": torch.backends.cudnn.version(),
        }
        if torch.cuda.is_available():
            payload["devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass
    return payload


def _resolve_project_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else Path(__file__).resolve().parents[1] / path


def _checkpoint_file_verification(
    model_path: Path | None, entries: Any
) -> list[dict[str, Any]]:
    if model_path is None or not isinstance(entries, list):
        return []
    result = []
    for entry in entries:
        if not isinstance(entry, dict) or "path" not in entry:
            continue
        path = model_path / str(entry["path"])
        observed_size = path.stat().st_size if path.is_file() else None
        result.append(
            {
                "path": str(entry["path"]),
                "exists": path.is_file(),
                "expected_bytes": entry.get("bytes"),
                "observed_bytes": observed_size,
                "size_matches": (
                    entry.get("bytes") is None or observed_size == int(entry["bytes"])
                ),
                "sha256": entry.get("sha256"),
            }
        )
    return result


def _checkpoint_fingerprint(
    model_path: Path | None, verified: list[dict[str, Any]]
) -> str | None:
    if verified and all(row.get("sha256") for row in verified):
        encoded = json.dumps(verified, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if model_path is None or not model_path.is_dir():
        return None
    inventory = [
        {"path": path.relative_to(model_path).as_posix(), "bytes": path.stat().st_size}
        for path in sorted(model_path.rglob("*"))
        if path.is_file() and ".cache" not in path.parts
    ]
    encoded = json.dumps(inventory, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_run(
    name: str,
    config: dict[str, Any],
    records: list[SceneRecord],
    rows: list[dict[str, Any]],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir or config["experiment"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    resolved_path = output / "config.resolved.yaml"
    resolved_path.write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    for row in rows:
        row["correct"] = bool(row["correct"])
    bootstrap_samples = int(config.get("evaluation", {}).get("bootstrap_samples", 1000))
    seed = int(config["experiment"]["seed"])
    metrics = evaluate_rows(rows, bootstrap_samples=bootstrap_samples, seed=seed)
    metrics["contrasts"] = summarize_interface_gap(
        rows, bootstrap_samples=bootstrap_samples, seed=seed
    )
    metrics["experiment"] = name
    metrics["model_backend"] = config.get("model", {}).get("backend", "reference")
    metrics["seed"] = seed
    metrics["dataset_size"] = len(records)
    write_metrics(output / "metrics.json", metrics)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "experiment": name,
        "status": "complete",
        "config_path": config.get("meta", {}).get("config_path"),
        "seed": seed,
        "model": dict(config.get("model", {})),
        "checkpoint": checkpoint_provenance(config),
        "dataset_size": len(records),
        "prediction_count": len(rows),
        "git_commit": git_commit(),
        "config_sha256": hashlib.sha256(resolved_path.read_bytes()).hexdigest(),
        "metrics_path": str((output / "metrics.json").as_posix()),
    }
    (output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "run.log").write_text(
        json.dumps(
            {"event": "complete", "experiment": name, "git_commit": manifest["git_commit"]}
        )
        + "\n",
        encoding="utf-8",
    )
    logging.getLogger(__name__).info("completed %s: %s", name, output)
    return metrics


def run_standard_suite(
    config: dict[str, Any],
    *,
    name: str,
    conditions: Iterable[str],
    count: int | None = None,
) -> dict[str, Any]:
    configure_logging()
    seed_everything(int(config["experiment"]["seed"]))
    generator = build_generator(config)
    output = Path(config["experiment"]["output_dir"])
    records = generator.generate(
        int(count or config.get("synthetic", {}).get("count", 30)),
        output_dir=output / "dataset",
        seed=int(config["experiment"]["seed"]),
    )
    vlm = build_vlm(config)
    return write_run(
        name,
        config,
        records,
        run_conditions(records, vlm, conditions),
        output_dir=output,
    )


def configured_conditions(
    config: dict[str, Any], default: Iterable[str] = CONDITIONS
) -> tuple[str, ...]:
    values = config.get("evaluation", {}).get("conditions")
    return tuple(str(value) for value in values) if values else tuple(default)
