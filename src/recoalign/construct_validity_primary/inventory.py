"""Byte-audited inheritance of the frozen PIVOT_EXP_A3 primary inventory."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
STUDY_ID = "PIVOT_EXP_A3P"
PARENT_STUDY_ID = "PIVOT_EXP_A3"
STUDY_ROOT = ROOT / "research/construct_validity/PIVOT_EXP_A3P"
PARENT_ROOT = ROOT / "research/construct_validity/PIVOT_EXP_A3"
PARENT_INVENTORY = PARENT_ROOT / "validation/trial_inventory.jsonl.gz"
INHERITED_INVENTORY = STUDY_ROOT / "validation/inherited_primary_inventory.jsonl.gz"
PARENT_ASSET_MANIFEST = PARENT_ROOT / "validation/visual_asset_manifest.jsonl"
DEVELOPMENT_INVENTORY = STUDY_ROOT / "validation/development_primary_inventory.jsonl.gz"

REGISTERED_CHOICE_IDS = ("1", "2", "3", "4")
PRIMARY_METHOD = "conditional_log_likelihood_single_token_argmax_v1"
PROTECTED_FIELDS = (
    "trial_key",
    "seed",
    "scene_id",
    "construct",
    "manipulation",
    "evidence_truth",
    "image_context",
    "task",
    "question",
    "choices",
    "scene_truth_answer",
    "declared_answer",
    "scoring_target",
    "correct_choice_id",
    "scene_truth_choice_id",
    "declared_choice_id",
    "prompt",
    "prompt_sha256",
    "image",
    "image_sha256",
    "scaffold_sha256",
)
EXPECTED_PRIMARY_ROWS = 27_000
EXPECTED_SEEDS = (20260901, 20260902, 20260903, 20260904, 20260905)

_MISSING = object()


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl_gz(path: str | Path) -> list[dict[str, Any]]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl_gz_atomic(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode(
            "utf-8"
        )
        for row in rows
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                compressed.write(payload)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary_name, destination)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def materialize_inherited_primary_inventory(*, allow_create: bool = True) -> dict[str, Any]:
    """Filter forced-choice rows while changing only the administrative study_id."""

    parent_rows = read_jsonl_gz(PARENT_INVENTORY)
    parent_primary = [row for row in parent_rows if row.get("response_method") == "forced_choice"]
    if len(parent_primary) != EXPECTED_PRIMARY_ROWS:
        raise ValueError(f"parent primary row count drift: {len(parent_primary)}")
    derived = [{**row, "study_id": STUDY_ID} for row in parent_primary]
    if INHERITED_INVENTORY.exists():
        existing = read_jsonl_gz(INHERITED_INVENTORY)
        if existing != derived:
            raise ValueError("existing inherited primary inventory differs from frozen derivation")
    elif allow_create:
        write_jsonl_gz_atomic(INHERITED_INVENTORY, derived)
    else:
        raise FileNotFoundError(INHERITED_INVENTORY)
    report = compare_primary_rows(parent_primary, derived)
    report.update(
        {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "parent_study_id": PARENT_STUDY_ID,
            "parent_inventory_sha256": sha256(PARENT_INVENTORY),
            "inherited_inventory_sha256": sha256(INHERITED_INVENTORY),
            "derivation": "filter_response_method_equals_forced_choice",
            "parent_rows_read": len(parent_rows),
            "secondary_rows_copied": 0,
            "validation_predictions_observed": False,
            "scientific_metrics_observed": False,
        }
    )
    asset_report = verify_primary_images(derived)
    report["image_asset_verification"] = asset_report
    report["passed"] = bool(report["passed"] and asset_report["passed"])
    return report


def compare_primary_rows(
    parent_rows: list[dict[str, Any]], inherited_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    exact = 0
    for index, (parent, inherited) in enumerate(zip(parent_rows, inherited_rows, strict=False)):
        row_mismatches = []
        for field in PROTECTED_FIELDS:
            parent_value = parent.get(field, _MISSING)
            inherited_value = inherited.get(field, _MISSING)
            if parent_value != inherited_value:
                row_mismatches.append(field)
        administrative_ok = (
            inherited.get("study_id") == STUDY_ID
            and {**inherited, "study_id": parent.get("study_id")} == parent
        )
        if row_mismatches or not administrative_ok:
            if len(mismatches) < 20:
                mismatches.append(
                    {
                        "row_index": index,
                        "trial_key": inherited.get("trial_key"),
                        "protected_fields": row_mismatches,
                        "administrative_only_change": administrative_ok,
                    }
                )
        else:
            exact += 1
    counts_match = len(parent_rows) == len(inherited_rows) == EXPECTED_PRIMARY_ROWS
    return {
        "expected_rows": EXPECTED_PRIMARY_ROWS,
        "actual_rows": len(inherited_rows),
        "parent_primary_rows": len(parent_rows),
        "exact_protected_field_matches": exact,
        "mismatch_count": len(inherited_rows) - exact
        if counts_match
        else abs(len(parent_rows) - len(inherited_rows)),
        "mismatches": mismatches,
        "protected_fields": list(PROTECTED_FIELDS),
        "image_sha256_field_policy": (
            "The frozen parent rows omit image_sha256. Absence is protected as an exact field "
            "state; image bytes are verified separately against the frozen asset manifest."
        ),
        "administrative_study_id_changed": True,
        "trial_keys_changed": False,
        "passed": counts_match and exact == EXPECTED_PRIMARY_ROWS and not mismatches,
    }


def parent_asset_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for row in read_jsonl(PARENT_ASSET_MANIFEST):
        path = _normal_path(str(row["path"]))
        digest = str(row["sha256"])
        if path in hashes and hashes[path] != digest:
            raise ValueError(f"conflicting asset hashes for {path}")
        hashes[path] = digest
    neutral = PARENT_ROOT / "semantic_manipulations/neutral.png"
    hashes[_normal_path(str(neutral))] = sha256(neutral)
    inventory_manifest_path = PARENT_ROOT / "validation/inventory_manifest.yaml"
    inventory_manifest = yaml.safe_load(inventory_manifest_path.read_text(encoding="utf-8"))
    for relative, metadata in inventory_manifest["files"].items():
        if not relative.endswith("/dataset.jsonl") or "/validation/" not in relative:
            continue
        dataset_path = ROOT / relative
        if sha256(dataset_path) != metadata["sha256"]:
            raise ValueError(f"frozen parent scene dataset hash drift: {relative}")
        for scene in read_jsonl(dataset_path):
            image = _normal_path(str(scene["image"]))
            digest = str(scene["metadata"]["image_sha256"])
            if image in hashes and hashes[image] != digest:
                raise ValueError(f"conflicting frozen scene image hash for {image}")
            hashes[image] = digest
    return hashes


def verify_primary_images(rows: list[dict[str, Any]]) -> dict[str, Any]:
    registered = parent_asset_hashes()
    unique_paths = sorted({_normal_path(str(row["image"])) for row in rows})
    missing_manifest: list[str] = []
    missing_files: list[str] = []
    hash_mismatches: list[str] = []
    for normalized in unique_paths:
        expected = registered.get(normalized)
        path = Path(normalized)
        if expected is None:
            missing_manifest.append(normalized)
            continue
        if not path.is_file():
            missing_files.append(normalized)
            continue
        if sha256(path) != expected:
            hash_mismatches.append(normalized)
    return {
        "unique_image_count": len(unique_paths),
        "manifest_registered_count": len(unique_paths) - len(missing_manifest),
        "verified_count": len(unique_paths)
        - len(missing_manifest)
        - len(missing_files)
        - len(hash_mismatches),
        "missing_manifest_entries": missing_manifest[:20],
        "missing_files": missing_files[:20],
        "hash_mismatches": hash_mismatches[:20],
        "passed": not missing_manifest and not missing_files and not hash_mismatches,
    }


def validate_inventory_shape(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [str(row.get("trial_key")) for row in rows]
    by_seed = Counter(int(row.get("seed", -1)) for row in rows)
    checks = {
        "row_count": len(rows) == EXPECTED_PRIMARY_ROWS,
        "unique_trial_keys": len(set(keys)) == len(keys) == EXPECTED_PRIMARY_ROWS,
        "five_complete_seeds": by_seed == Counter({seed: 5400 for seed in EXPECTED_SEEDS}),
        "forced_choice_only": all(row.get("response_method") == "forced_choice" for row in rows),
        "prompt_suffix": all(str(row.get("prompt", "")).endswith("FINAL_CHOICE=") for row in rows),
        "prompt_hashes": all(
            hashlib.sha256(str(row["prompt"]).encode("utf-8")).hexdigest()
            == row.get("prompt_sha256")
            for row in rows
        ),
        "registered_choices": all(
            tuple(row.get("choices", ())) == REGISTERED_CHOICE_IDS for row in rows
        ),
        "registered_correct_mappings": all(
            str(row.get(field)) in REGISTERED_CHOICE_IDS
            for row in rows
            for field in ("correct_choice_id", "scene_truth_choice_id", "declared_choice_id")
        ),
        "known_constructs": {str(row.get("construct")) for row in rows} == {"CV1", "CV2", "CV3"},
        "known_manipulations": {str(row.get("manipulation")) for row in rows} == {"M0", "M1", "M2"},
    }
    return {
        "checks": checks,
        "by_seed": dict(sorted(by_seed.items())),
        "passed": all(checks.values()),
    }


def load_development_inventory() -> list[dict[str, Any]]:
    rows = read_jsonl_gz(DEVELOPMENT_INVENTORY)
    checks = {
        "row_count": len(rows) == 432,
        "development_seed": {int(row.get("seed", -1)) for row in rows} == {20260830},
        "unique_keys": len({str(row.get("trial_key")) for row in rows}) == 432,
        "forced_choice_only": all(row.get("response_method") == "forced_choice" for row in rows),
        "prompt_suffix": all(str(row.get("prompt", "")).endswith("FINAL_CHOICE=") for row in rows),
        "registered_choices": all(
            tuple(row.get("choices", ())) == REGISTERED_CHOICE_IDS for row in rows
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"development primary inventory drift: {checks}")
    return rows


def write_inventory_reports(report: dict[str, Any]) -> None:
    comparison = STUDY_ROOT / "validation/parent_inventory_comparison.yaml"
    comparison.parent.mkdir(parents=True, exist_ok=True)
    comparison.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    image = report["image_asset_verification"]
    markdown = f"""# PIVOT_EXP_A3P Primary Immutability Report

PIVOT_EXP_A3P inherits only the frozen forced-choice rows from PIVOT_EXP_A3. The
administrative `study_id` is the sole row-level change; trial keys remain the parent keys.

- Expected rows: {report["expected_rows"]:,}
- Actual rows: {report["actual_rows"]:,}
- Exact protected-field matches: {report["exact_protected_field_matches"]:,}
- Mismatches: {report["mismatch_count"]}
- Unique image assets verified: {image["verified_count"]:,}
- Parent inventory SHA-256: `{report["parent_inventory_sha256"]}`
- Inherited inventory SHA-256: `{report["inherited_inventory_sha256"]}`

The parent inventory does not materialize an `image_sha256` field in individual rows. That
absence is itself protected and matches in all 27,000 comparisons. Image bytes are independently
checked against the frozen parent visual-asset manifest. No scene, image, legend, question,
choice, answer mapping, scaffold, prompt, prompt hash, corruption, or trial key was regenerated or
rewritten.
"""
    (STUDY_ROOT / "validation/primary_immutability_report.md").write_text(
        markdown, encoding="utf-8"
    )


def _normal_path(value: str) -> str:
    return str(Path(value).resolve())


__all__ = [
    "DEVELOPMENT_INVENTORY",
    "EXPECTED_PRIMARY_ROWS",
    "EXPECTED_SEEDS",
    "INHERITED_INVENTORY",
    "PRIMARY_METHOD",
    "PROTECTED_FIELDS",
    "REGISTERED_CHOICE_IDS",
    "compare_primary_rows",
    "load_development_inventory",
    "materialize_inherited_primary_inventory",
    "parent_asset_hashes",
    "read_jsonl_gz",
    "sha256",
    "validate_inventory_shape",
    "write_inventory_reports",
    "write_jsonl_gz_atomic",
]
