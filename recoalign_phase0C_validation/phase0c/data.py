from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_registered_records(config: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index_path = config.source_features_dir / "index.jsonl"
    selection_path = config.source_phase0b_dir / "configs" / "selection_manifest.json"
    llm_manifest_path = config.source_features_dir / "llm_manifest.json"
    vision_manifest_path = config.source_features_dir / "vision_manifest.json"
    for path in (index_path, selection_path, llm_manifest_path, vision_manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Required completed Phase 0-B artifact missing: {path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    llm_manifest = json.loads(llm_manifest_path.read_text(encoding="utf-8"))
    vision_manifest = json.loads(vision_manifest_path.read_text(encoding="utf-8"))
    observed_hash = sha256_file(index_path)
    if selection.get("selected_index_sha256") != observed_hash:
        raise RuntimeError("Phase 0-B selected index hash does not match its manifest")
    if llm_manifest.get("index_sha256") != observed_hash or not llm_manifest.get("complete"):
        raise RuntimeError("Phase 0-B LLM feature manifest is incomplete or mismatched")
    if vision_manifest.get("index_sha256") != observed_hash or not vision_manifest.get("complete"):
        raise RuntimeError("Phase 0-B vision feature manifest is incomplete or mismatched")
    records = read_jsonl(index_path)
    if len(records) != config.n_train + config.n_test:
        raise RuntimeError(f"Expected 1008 registered rows, observed {len(records)}")
    if sum(row["split"] == "train" for row in records) != config.n_train:
        raise RuntimeError("Registered training split size changed")
    if sum(row["split"] == "test" for row in records) != config.n_test:
        raise RuntimeError("Registered test split size changed")
    return records, {
        "index_path": str(index_path.resolve()),
        "index_sha256": observed_hash,
        "selection_manifest_sha256": sha256_file(selection_path),
        "llm_manifest_sha256": sha256_file(llm_manifest_path),
        "vision_manifest_sha256": sha256_file(vision_manifest_path),
        "za_tokens_path": str((config.source_features_dir / "za_tokens.npy").resolve()),
    }
