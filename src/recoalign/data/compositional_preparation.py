"""Prepare local ARO, Winoground, and BiVLC snapshots without redistributing data."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from recoalign.benchmarks.caption_multisets import (
    WHITESPACE_TOKEN_MULTISET,
    WINOGROUND_ALPHANUMERIC_CHARACTER_MULTISET,
    caption_multiset_matches,
)
from recoalign.data.manifest import sha256_file


def prepare_aro(
    source_jsonl: str | Path,
    dataset_root: str | Path,
    *,
    manifest_output: str | Path,
    source: str,
    license_name: str,
    hash_images: bool = False,
) -> dict[str, Any]:
    """Validate and normalize a path-based export of the four ARO subsets."""
    source = _nonempty_argument(source, "source")
    license_name = _nonempty_argument(license_name, "license_name")
    root, image_root, source_dir, annotation_dir, inventory_dir = _prepare_directories(
        dataset_root, "aro"
    )
    source_copy = source_dir / "aro.jsonl"
    _copy_source(source_jsonl, source_copy)

    rows: list[dict[str, Any]] = []
    inventory_paths: set[Path] = set()
    subset_counts: Counter[str] = Counter()
    seen: set[str] = set()
    for index, row in enumerate(_load_jsonl(source_copy), start=1):
        sample_id = _required_text(row, "sample_id", index)
        if sample_id in seen:
            raise ValueError(f"ARO row {index}: duplicate sample_id {sample_id!r}")
        seen.add(sample_id)
        relative_image = _safe_relative_path(_required_text(row, "image", index), index)
        _require_image(image_root, relative_image, "ARO")

        captions = row.get("captions")
        if not isinstance(captions, list) or len(captions) < 2:
            raise ValueError(f"ARO row {index}: captions must contain at least two strings")
        normalized_captions = [
            _nonempty_text(caption, f"ARO row {index}: caption") for caption in captions
        ]
        correct_index = row.get("correct_index")
        if (
            not isinstance(correct_index, int)
            or isinstance(correct_index, bool)
            or not 0 <= correct_index < len(normalized_captions)
        ):
            raise ValueError(f"ARO row {index}: correct_index must reference captions")
        subset = _required_text(row, "subset", index)
        tags = _optional_string_list(row.get("tags"), f"ARO row {index}: tags")
        metadata = _optional_metadata(row.get("metadata"), f"ARO row {index}: metadata")

        rows.append(
            {
                "sample_id": sample_id,
                "image": relative_image.as_posix(),
                "captions": normalized_captions,
                "correct_index": correct_index,
                "subset": subset,
                "tags": tags,
                "metadata": metadata,
            }
        )
        subset_counts[subset] += 1
        inventory_paths.add(Path("images") / relative_image)

    if not rows:
        raise ValueError("ARO source contains no records")
    return _write_snapshot(
        root,
        source_copy,
        rows,
        inventory_paths,
        annotation_dir,
        inventory_dir,
        manifest_output=manifest_output,
        name="aro",
        version="official-export-v1",
        source=source,
        license_name=license_name,
        processing={
            "format": "recoalign-multichoice-jsonl-v1",
            "subsets": dict(sorted(subset_counts.items())),
        },
        notes=(
            "Path-based export of Visual Genome Attribution/Relation and COCO/Flickr30K Order. "
            "The official benchmark assets remain subject to their upstream terms."
        ),
        hash_images=hash_images,
    )


def prepare_winoground(
    source_jsonl: str | Path,
    dataset_root: str | Path,
    *,
    manifest_output: str | Path,
    source: str,
    license_name: str,
    hash_images: bool = False,
    source_revision: str | None = None,
    exporter_version: str | None = None,
    downloaded_at: str | None = None,
) -> dict[str, Any]:
    """Prepare the official 400-example Winoground export."""
    return _prepare_paired_matrix(
        source_jsonl,
        dataset_root,
        manifest_output=manifest_output,
        source=source,
        license_name=license_name,
        dataset_name="winoground",
        version="official-400-v1",
        default_category="winoground",
        notes=(
            "Each row contains two images and two captions with diagonal ground-truth matches. "
            "The recorded content check compares case-folded alphanumeric character frequencies "
            "without changing caption text; it is not linguistic tokenization or segmentation."
        ),
        hash_images=hash_images,
        source_revision=source_revision,
        exporter_version=exporter_version,
        downloaded_at=_rfc3339_utc(downloaded_at),
    )


def prepare_bivlc(
    source_jsonl: str | Path,
    dataset_root: str | Path,
    *,
    manifest_output: str | Path,
    source: str,
    license_name: str,
    hash_images: bool = False,
) -> dict[str, Any]:
    """Prepare a path-based export of the human-filtered BiVLC benchmark."""
    return _prepare_paired_matrix(
        source_jsonl,
        dataset_root,
        manifest_output=manifest_output,
        source=source,
        license_name=license_name,
        dataset_name="bivlc",
        version="official-human-filtered-v1",
        default_category="bivlc",
        notes=(
            "Each row contains the positive and synthetic-negative image/caption pair with "
            "diagonal ground-truth matches."
        ),
        hash_images=hash_images,
    )


def _prepare_paired_matrix(
    source_jsonl: str | Path,
    dataset_root: str | Path,
    *,
    manifest_output: str | Path,
    source: str,
    license_name: str,
    dataset_name: str,
    version: str,
    default_category: str,
    notes: str,
    hash_images: bool,
    source_revision: str | None = None,
    exporter_version: str | None = None,
    downloaded_at: str | None = None,
) -> dict[str, Any]:
    source = _nonempty_argument(source, "source")
    license_name = _nonempty_argument(license_name, "license_name")
    root, image_root, source_dir, annotation_dir, inventory_dir = _prepare_directories(
        dataset_root, dataset_name
    )
    source_copy = source_dir / f"{dataset_name}.jsonl"
    _copy_source(source_jsonl, source_copy)

    rows: list[dict[str, Any]] = []
    inventory_paths: set[Path] = set()
    category_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    seen: set[str] = set()
    content_check_matches = 0
    content_check_method = (
        WINOGROUND_ALPHANUMERIC_CHARACTER_MULTISET
        if dataset_name == "winoground"
        else WHITESPACE_TOKEN_MULTISET
    )
    for index, row in enumerate(_load_jsonl(source_copy), start=1):
        sample_id = _required_text(row, "sample_id", index)
        if sample_id in seen:
            raise ValueError(
                f"{dataset_name} row {index}: duplicate sample_id {sample_id!r}"
            )
        seen.add(sample_id)
        image_0 = _safe_relative_path(_required_text(row, "image_0", index), index)
        image_1 = _safe_relative_path(_required_text(row, "image_1", index), index)
        _require_image(image_root, image_0, dataset_name)
        _require_image(image_root, image_1, dataset_name)
        caption_0 = _required_text(row, "caption_0", index)
        caption_1 = _required_text(row, "caption_1", index)
        category_value = row.get("category", default_category)
        category = _nonempty_text(category_value, f"{dataset_name} row {index}: category")
        tags = _optional_string_list(row.get("tags"), f"{dataset_name} row {index}: tags")
        metadata = _optional_metadata(
            row.get("metadata"), f"{dataset_name} row {index}: metadata"
        )

        rows.append(
            {
                "sample_id": sample_id,
                "image_0": image_0.as_posix(),
                "image_1": image_1.as_posix(),
                "caption_0": caption_0,
                "caption_1": caption_1,
                "category": category,
                "tags": tags,
                "metadata": metadata,
            }
        )
        inventory_paths.update((Path("images") / image_0, Path("images") / image_1))
        category_counts[category] += 1
        tag_counts.update(tags)
        content_check_matches += caption_multiset_matches(
            caption_0,
            caption_1,
            method=content_check_method,
        )

    if not rows:
        raise ValueError(f"{dataset_name} source contains no records")
    content_check_rate = 100.0 * content_check_matches / len(rows)
    content_check_processing: dict[str, Any]
    if dataset_name == "winoground":
        content_check_processing = {
            "caption_alphanumeric_character_multiset_match_rate": content_check_rate,
            "caption_alphanumeric_character_multiset_method": content_check_method,
            # Deprecated aliases retained for existing configs and downstream readers.
            "caption_token_multiset_match_rate": content_check_rate,
            "caption_token_multiset_method": content_check_method,
            **_winoground_provenance(
                rows,
                source_revision=source_revision,
                exporter_version=exporter_version,
                downloaded_at=downloaded_at,
            ),
        }
    else:
        content_check_processing = {
            "caption_token_multiset_match_rate": content_check_rate,
            "caption_token_multiset_method": content_check_method,
        }
    return _write_snapshot(
        root,
        source_copy,
        rows,
        inventory_paths,
        annotation_dir,
        inventory_dir,
        manifest_output=manifest_output,
        name=dataset_name,
        version=version,
        source=source,
        license_name=license_name,
        processing={
            "format": "recoalign-paired-matrix-jsonl-v1",
            "categories": dict(sorted(category_counts.items())),
            "tags": dict(sorted(tag_counts.items())),
            **content_check_processing,
        },
        notes=notes,
        hash_images=hash_images,
        downloaded_at=downloaded_at,
    )


def _write_snapshot(
    root: Path,
    source_copy: Path,
    rows: list[dict[str, Any]],
    inventory_paths: set[Path],
    annotation_dir: Path,
    inventory_dir: Path,
    *,
    manifest_output: str | Path,
    name: str,
    version: str,
    source: str,
    license_name: str,
    processing: dict[str, Any],
    notes: str,
    hash_images: bool,
    downloaded_at: str | None = None,
) -> dict[str, Any]:
    annotation_path = annotation_dir / "test.jsonl"
    _write_jsonl(annotation_path, rows)
    inventory_path = inventory_dir / "images.jsonl"
    _write_inventory(root, sorted(inventory_paths), inventory_path, hash_images=hash_images)
    processing = {
        **processing,
        "source_annotation": f"source/{source_copy.name}",
        "image_inventory": "inventories/images.jsonl",
        "inventory_splits": ["test"],
        "image_hashes": hash_images,
    }
    manifest = {
        "schema_version": 1,
        "name": name,
        "version": version,
        "source": source,
        "license": license_name,
        "downloaded_at": downloaded_at,
        "splits": {"test": len(rows)},
        "files": [
            _manifest_entry(source_copy, root),
            _manifest_entry(annotation_path, root),
            _manifest_entry(inventory_path, root),
        ],
        "processing": processing,
        "notes": notes,
    }
    _write_manifest(manifest_output, manifest)
    return manifest


def _prepare_directories(
    dataset_root: str | Path,
    dataset_name: str,
) -> tuple[Path, Path, Path, Path, Path]:
    root = Path(dataset_root)
    image_root = root / "images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"{dataset_name} image directory does not exist: {image_root}")
    source_dir = root / "source"
    annotation_dir = root / "annotations"
    inventory_dir = root / "inventories"
    source_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    inventory_dir.mkdir(parents=True, exist_ok=True)
    return root, image_root, source_dir, annotation_dir, inventory_dir


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"row {line_number}: JSONL value must be an object")
            rows.append(payload)
    return rows


def _write_inventory(
    root: Path,
    relative_paths: list[Path],
    output: Path,
    *,
    hash_images: bool,
) -> None:
    rows: list[dict[str, Any]] = []
    for relative_path in relative_paths:
        path = root / relative_path
        row: dict[str, Any] = {
            "path": relative_path.as_posix(),
            "bytes": path.stat().st_size,
        }
        if hash_images:
            row["sha256"] = sha256_file(path)
        rows.append(row)
    _write_jsonl(output, rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False, allow_unicode=True)


def _manifest_entry(path: Path, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root)
    return {
        "path": relative.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _copy_source(source: str | Path, destination: Path) -> None:
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"source annotation does not exist: {source_path}")
    if source_path.resolve() != destination.resolve():
        shutil.copyfile(source_path, destination)


def _safe_relative_path(value: str, index: int) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"row {index}: unsafe image path")
    return path


def _require_image(image_root: Path, relative_path: Path, dataset_name: str) -> None:
    path = image_root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"{dataset_name} image is missing: {path}")


def _required_text(mapping: dict[str, Any], key: str, index: int) -> str:
    if key not in mapping:
        raise ValueError(f"row {index}: missing field {key!r}")
    return _nonempty_text(mapping[key], f"row {index}: {key}")


def _optional_string_list(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of non-empty strings")
    normalized = [_nonempty_text(item, label) for item in value]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} must not contain duplicates")
    return normalized


def _optional_metadata(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _nonempty_argument(value: str, label: str) -> str:
    return _nonempty_text(value, label)


def _winoground_provenance(
    rows: list[dict[str, Any]],
    *,
    source_revision: str | None,
    exporter_version: str | None,
    downloaded_at: str | None,
) -> dict[str, Any]:
    formal_snapshot = len(rows) == 400
    if formal_snapshot:
        row_dataset = _required_consistent_metadata_value(rows, "source_dataset")
        row_split = _required_consistent_metadata_value(rows, "source_split")
        revision = _required_consistent_metadata_value(rows, "source_revision")
        exporter = _required_consistent_metadata_value(rows, "exporter_version")
        _validate_explicit_provenance(source_revision, revision, "source_revision")
        _validate_explicit_provenance(exporter_version, exporter, "exporter_version")
        if row_dataset != "facebook/winoground":
            raise ValueError(
                "formal Winoground snapshot source_dataset must be facebook/winoground"
            )
        if row_split != "test":
            raise ValueError("formal Winoground snapshot source_split must be test")
        if re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
            raise ValueError(
                "formal Winoground snapshot source_revision must be a 40-character commit SHA"
            )
        if downloaded_at is None:
            raise ValueError("formal Winoground snapshot requires downloaded_at")
        provenance_status = "pinned_revision_verified"
    else:
        row_dataset = _consistent_metadata_value(rows, "source_dataset")
        row_split = _consistent_metadata_value(rows, "source_split")
        row_revision = _consistent_metadata_value(rows, "source_revision")
        row_exporter = _consistent_metadata_value(rows, "exporter_version")
        revision = _resolve_optional_provenance(
            source_revision, row_revision, "source_revision"
        )
        exporter = _resolve_optional_provenance(
            exporter_version, row_exporter, "exporter_version"
        )
        if revision is not None and re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
            raise ValueError("source_revision must be a 40-character commit SHA")
        provenance_status = "synthetic_or_unverified"
    return {
        "source_dataset": row_dataset,
        "source_split": row_split,
        "source_revision": revision,
        "exporter_version": exporter,
        "provenance_status": provenance_status,
    }


def _consistent_metadata_value(rows: list[dict[str, Any]], field: str) -> str | None:
    values = [row["metadata"].get(field) for row in rows]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"Winoground rows disagree on metadata.{field}")
    value = values[0]
    if value is None:
        return None
    return _nonempty_text(value, f"Winoground metadata.{field}")


def _required_consistent_metadata_value(rows: list[dict[str, Any]], field: str) -> str:
    if any(field not in row["metadata"] or row["metadata"][field] is None for row in rows):
        if field == "source_revision":
            raise ValueError("formal Winoground snapshot is missing pinned row source_revision")
        if field == "exporter_version":
            raise ValueError("formal Winoground snapshot is missing row exporter_version")
        raise ValueError(f"formal Winoground snapshot is missing row metadata.{field}")
    value = _consistent_metadata_value(rows, field)
    if value is None:
        raise ValueError(f"formal Winoground snapshot is missing row metadata.{field}")
    return value


def _validate_explicit_provenance(
    explicit: str | None,
    recorded: str,
    field: str,
) -> None:
    if explicit is None:
        return
    normalized = _nonempty_argument(explicit, field)
    if normalized != recorded:
        raise ValueError(f"{field} does not match exported row metadata")


def _resolve_optional_provenance(
    explicit: str | None,
    recorded: str | None,
    field: str,
) -> str | None:
    normalized = None if explicit is None else _nonempty_argument(explicit, field)
    if normalized is not None and recorded is not None and normalized != recorded:
        raise ValueError(f"{field} does not match exported row metadata")
    return normalized or recorded


def _rfc3339_utc(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _nonempty_argument(value, "downloaded_at")
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("downloaded_at must be an RFC 3339 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("downloaded_at must be an RFC 3339 UTC timestamp")
    return parsed.isoformat().replace("+00:00", "Z")
