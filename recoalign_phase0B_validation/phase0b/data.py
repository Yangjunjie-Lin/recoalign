from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import ExperimentConfig


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _balanced_sample(
    records: list[dict[str, Any]], *, split: str, count: int, seed: int
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["split"] == split:
            groups[record["composition"]].append(record)
    if len(groups) != 72:
        raise RuntimeError(f"Expected 72 {split} composition classes, found {len(groups)}")
    per_class = count // 72
    selected: list[dict[str, Any]] = []
    for class_index, label in enumerate(sorted(groups)):
        candidates = sorted(groups[label], key=lambda row: row["image_id"])
        rng = random.Random(seed + class_index * 104729)
        rng.shuffle(candidates)
        if len(candidates) < per_class:
            raise RuntimeError(f"Class {label!r} has only {len(candidates)} {split} rows")
        selected.extend(candidates[:per_class])
    random.Random(seed + 99991).shuffle(selected)
    return selected


def select_protocol_records(config: ExperimentConfig) -> list[dict[str, Any]]:
    """Select a fixed, class-balanced subset from the completed Phase 0-A dataset."""

    config.validate()
    source_path = config.source_dataset_dir / "metadata.jsonl"
    source_manifest_path = config.source_dataset_dir / "manifest.json"
    if not source_path.exists() or not source_manifest_path.exists():
        raise FileNotFoundError("Phase 0-A synthetic dataset is required before Phase 0-B can run")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_hash = sha256_file(source_path)
    if source_manifest.get("metadata_sha256") != source_hash:
        raise RuntimeError("Phase 0-A metadata hash does not match its manifest")

    records = read_jsonl(source_path)
    selected = _balanced_sample(
        records, split="train", count=config.n_train, seed=config.seed + 101
    ) + _balanced_sample(records, split="test", count=config.n_test, seed=config.seed + 211)
    for row, record in enumerate(selected):
        record["phase0b_row"] = row
        record["absolute_image_path"] = str((config.source_dataset_dir / record["image"]).resolve())

    index_path = config.features_dir / "index.jsonl"
    write_jsonl(index_path, selected)
    selection_manifest = {
        "seed": config.seed,
        "n_train": config.n_train,
        "n_test": config.n_test,
        "classes": 72,
        "samples_per_train_class": config.n_train // 72,
        "samples_per_test_class": config.n_test // 72,
        "source_metadata": str(source_path.resolve()),
        "source_metadata_sha256": source_hash,
        "selected_index_sha256": sha256_file(index_path),
    }
    config.configs_dir.mkdir(parents=True, exist_ok=True)
    (config.configs_dir / "selection_manifest.json").write_text(
        json.dumps(selection_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return selected


def labels_for_split(records: list[dict[str, Any]], split: str, semantic: str) -> list[str]:
    return [row["semantic_label"][semantic] for row in records if row["split"] == split]
