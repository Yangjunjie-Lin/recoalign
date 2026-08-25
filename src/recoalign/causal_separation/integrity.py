"""Immutable-parent, design, prediction, and artifact integrity gates."""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


def validate_parent_freeze(root: Path, freeze_record: str | Path) -> dict[str, Any]:
    path = root / freeze_record
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("PIVOT_EXP_A final freeze record must be a mapping")
    asset_commit = str(payload["asset_commit_sha"])
    assertions: dict[str, bool] = {
        "decision_no_go": payload["decision"] == "NO-GO",
        "prediction_lines_2000": _gzip_lines(root / payload["immutability"]["predictions"])
        == 2000,
        "freeze_sha256": _sha256(root / "research/pivot_validation/freeze_manifest.yaml")
        == payload["freeze_sha256"],
        "predictions_sha256": _sha256(root / payload["immutability"]["predictions"])
        == payload["predictions_sha256"],
    }
    immutable_paths: list[str] = []
    for key, value in payload["immutability"].items():
        if key == "mutation_allowed":
            continue
        immutable_paths.extend(value if isinstance(value, list) else [value])
    blob_match = {}
    for relative in immutable_paths:
        current = (root / relative).read_bytes()
        committed = subprocess.run(
            ["git", "show", f"{asset_commit}:{relative}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        blob_match[relative] = hashlib.sha256(current).digest() == hashlib.sha256(
            committed
        ).digest()
    assertions["all_parent_assets_match_frozen_commit"] = all(blob_match.values())
    return {
        "passed": all(assertions.values()),
        "assertions": assertions,
        "asset_commit_sha": asset_commit,
        "blob_match": blob_match,
        "freeze_record_sha256": _sha256(path),
    }


def validate_prediction_rows(
    rows: list[dict[str, Any]], config: dict[str, Any], freeze: dict[str, Any]
) -> dict[str, Any]:
    seeds = [int(seed) for seed in config["study"]["seeds"]]
    expected_per_seed = int(config["study"]["predictions_per_seed"])
    expected = expected_per_seed * len(seeds)
    by_seed = Counter(int(row["seed"]) for row in rows)
    main_by_scene: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["family"] == "main":
            main_by_scene[(int(row["seed"]), str(row["source_scene_id"]))].append(row)
    assertions = {
        "prediction_count": len(rows) == expected,
        "unique_trial_keys": len({row["trial_key"] for row in rows}) == len(rows),
        "five_complete_seeds": all(by_seed[seed] == expected_per_seed for seed in seeds),
        "complete_eight_cell_matrix": len(main_by_scene) == 450
        and all(len(scene_rows) == 8 for scene_rows in main_by_scene.values()),
        "all_rows_frozen": all(row["freeze_name"] == freeze["freeze_name"] for row in rows),
        "token_matched": all(
            max(int(row["input"]["input_tokens"]) for row in scene_rows)
            - min(int(row["input"]["input_tokens"]) for row in scene_rows)
            <= int(config["factorial_design"]["token_match_tolerance"])
            for scene_rows in main_by_scene.values()
        ),
        "corruption_validity": all(
            bool(row["metadata"]["semantic_corruption_false"])
            and bool(row["metadata"]["relation_corruption_false"])
            for row in rows
            if row["family"] == "main"
        ),
        "no_empty_predictions": all(str(row["prediction"]).strip() for row in rows),
        "all_predictions_parsed": all(
            row["evaluation"].get("matched_choice") is not None for row in rows
        ),
    }
    return {
        "passed": all(assertions.values()),
        "assertions": assertions,
        "expected": expected,
        "by_seed": dict(sorted(by_seed.items())),
    }


def locked_files_match(root: Path, freeze: dict[str, Any]) -> bool:
    return all(
        _sha256(root / relative) == digest
        for relative, digest in freeze["locked_files"].items()
    )


def source_files_match(root: Path, freeze: dict[str, Any]) -> bool:
    return all(
        _sha256(root / relative) == digest
        for relative, digest in freeze["source_datasets"].items()
    )


def validate_artifact_manifest(result_dir: Path) -> dict[str, Any]:
    manifest_path = result_dir / "artifact_manifest.yaml"
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assertions = {
        name: (result_dir / name).stat().st_size == int(metadata["bytes"])
        and _sha256(result_dir / name) == metadata["sha256"]
        for name, metadata in payload["artifacts"].items()
    }
    predictions = result_dir / "predictions.jsonl.gz"
    assertions["prediction_lines"] = _gzip_lines(predictions) == int(
        payload["prediction_lines"]
    )
    return {"passed": all(assertions.values()), "assertions": assertions}


def artifact_metadata(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _gzip_lines(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "artifact_metadata",
    "canonical_json_sha256",
    "locked_files_match",
    "source_files_match",
    "validate_artifact_manifest",
    "validate_parent_freeze",
    "validate_prediction_rows",
]
