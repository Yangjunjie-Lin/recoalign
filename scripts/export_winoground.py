#!/usr/bin/env python3
"""Export the gated official Winoground Hugging Face dataset to local files."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from recoalign.benchmarks.caption_multisets import (
    WINOGROUND_CONTENT_MULTISET,
    caption_multiset_matches,
)

DEFAULT_DATASET_ID = "facebook/winoground"
DEFAULT_SPLIT = "test"
EXPECTED_SAMPLES = 400
TAG_FIELDS = ("tag", "secondary_tag", "collapsed_tag")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export official Winoground rows without changing image-caption ordering."
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output-root", default="data/winoground")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--image-format", choices=("png", "jpeg"), default="png")
    return parser.parse_args()


def load_official_split(dataset_id: str, split: str) -> Any:
    """Load one split lazily so importing this script does not require datasets."""
    from datasets import load_dataset

    return load_dataset(dataset_id, split=split)


def export_winoground(
    rows: Sequence[Mapping[str, Any]],
    output_root: str | Path,
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    split: str = DEFAULT_SPLIT,
    dry_run: bool = False,
    overwrite: bool = False,
    max_samples: int | None = None,
    image_format: str = "png",
    expected_samples: int = EXPECTED_SAMPLES,
) -> dict[str, Any]:
    """Validate and export rows while preserving official field order and text."""
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if image_format not in {"png", "jpeg"}:
        raise ValueError("image_format must be 'png' or 'jpeg'")

    source_count = len(rows)
    if max_samples is None and source_count != expected_samples:
        raise ValueError(
            f"formal Winoground export requires {expected_samples} samples; observed {source_count}"
        )
    export_count = source_count if max_samples is None else min(source_count, max_samples)
    if export_count == 0:
        raise ValueError("Winoground split contains no rows")

    root = Path(output_root)
    if dry_run:
        summary = _validate_rows(
            rows,
            export_count,
            dataset_id=dataset_id,
            split=split,
            image_format=image_format,
        )
        return {**summary, "dry_run": True, "formal_export": max_samples is None}

    root.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(tempfile.mkdtemp(prefix=".winoground-export-", dir=root.parent))
    try:
        summary = _stage_rows(
            rows,
            export_count,
            staging_path,
            dataset_id=dataset_id,
            split=split,
            image_format=image_format,
        )
        filenames = [
            name
            for row in summary.pop("_rows")
            for name in (row["image_0"], row["image_1"])
        ]
        _install_staged_export(staging_path, root, filenames, overwrite=overwrite)
    finally:
        shutil.rmtree(staging_path, ignore_errors=True)

    return {**summary, "dry_run": False, "formal_export": max_samples is None}


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    count: int,
    *,
    dataset_id: str,
    split: str,
    image_format: str,
) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for index in range(count):
        record, _, _ = _normalize_row(
            rows[index], index, dataset_id=dataset_id, split=split, image_format=image_format
        )
        official_id = record["metadata"]["official_id"]
        if official_id in seen_ids:
            raise ValueError(f"row {index}: duplicate official id {official_id}")
        seen_ids.add(official_id)
        normalized.append(record)
    return _summary(normalized)


def _stage_rows(
    rows: Sequence[Mapping[str, Any]],
    count: int,
    staging_root: Path,
    *,
    dataset_id: str,
    split: str,
    image_format: str,
) -> dict[str, Any]:
    image_root = staging_root / "images"
    image_root.mkdir(parents=True)
    normalized: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for index in range(count):
        record, image_0, image_1 = _normalize_row(
            rows[index], index, dataset_id=dataset_id, split=split, image_format=image_format
        )
        official_id = record["metadata"]["official_id"]
        if official_id in seen_ids:
            raise ValueError(f"row {index}: duplicate official id {official_id}")
        seen_ids.add(official_id)
        _save_image(image_0, image_root / record["image_0"], image_format)
        _save_image(image_1, image_root / record["image_1"], image_format)
        normalized.append(record)

    incoming = staging_root / "incoming" / "winoground.jsonl"
    _atomic_write_jsonl(incoming, normalized)
    return {**_summary(normalized), "_rows": normalized}


def _normalize_row(
    row: Mapping[str, Any],
    index: int,
    *,
    dataset_id: str,
    split: str,
    image_format: str,
) -> tuple[dict[str, Any], Image.Image, Image.Image]:
    if not isinstance(row, Mapping):
        raise ValueError(f"row {index}: expected an object")
    official_id = _official_id(row, index)
    caption_0 = _required_caption(row, "caption_0", index)
    caption_1 = _required_caption(row, "caption_1", index)
    image_0 = _required_image(row, "image_0", index)
    image_1 = _required_image(row, "image_1", index)
    tags, tag_metadata = _official_tags(row, index)
    suffix = "png" if image_format == "png" else "jpg"
    stable_id = f"{official_id:06d}"
    metadata: dict[str, Any] = {
        "official_id": official_id,
        "source_dataset": dataset_id,
        "source_split": split,
    }
    if tag_metadata:
        metadata["official_tag_fields"] = tag_metadata
    return (
        {
            "sample_id": f"winoground-{stable_id}",
            "image_0": f"{stable_id}_image_0.{suffix}",
            "image_1": f"{stable_id}_image_1.{suffix}",
            "caption_0": caption_0,
            "caption_1": caption_1,
            "category": "winoground",
            "tags": tags,
            "metadata": metadata,
        },
        image_0,
        image_1,
    )


def _official_id(row: Mapping[str, Any], index: int) -> int:
    if "id" not in row:
        raise ValueError(f"row {index}: missing field 'id'")
    value = row["id"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"row {index}: id must be a non-negative integer")
    return value


def _required_caption(row: Mapping[str, Any], field: str, index: int) -> str:
    if field not in row:
        raise ValueError(f"row {index}: missing field {field!r}")
    value = row[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row {index}: {field} must be a non-empty string")
    return value


def _required_image(row: Mapping[str, Any], field: str, index: int) -> Image.Image:
    if field not in row:
        raise ValueError(f"row {index}: missing field {field!r}")
    value = row[field]
    if not isinstance(value, Image.Image):
        raise ValueError(f"row {index}: {field} must be a decoded Pillow image")
    return value


def _official_tags(
    row: Mapping[str, Any], index: int
) -> tuple[list[str], dict[str, str | list[str]]]:
    if "tags" in row:
        value = row["tags"]
        if not isinstance(value, list):
            raise ValueError(f"row {index}: tags must be a list of non-empty strings")
        return [_tag_value(item, "tags", index) for item in value], {}

    tags: list[str] = []
    metadata: dict[str, str | list[str]] = {}
    for field in TAG_FIELDS:
        if field not in row or row[field] is None:
            continue
        value = row[field]
        if isinstance(value, str):
            metadata[field] = value
            if value.strip():
                tags.append(value)
        elif isinstance(value, list):
            normalized = [_tag_value(item, field, index) for item in value]
            tags.extend(normalized)
            metadata[field] = normalized
        else:
            raise ValueError(f"row {index}: {field} must be a string or list of strings")
    return tags, metadata


def _tag_value(value: Any, field: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row {index}: {field} must contain non-empty strings")
    return value


def _save_image(image: Image.Image, path: Path, image_format: str) -> None:
    target_format = "PNG" if image_format == "png" else "JPEG"
    source = image if target_format == "PNG" or image.mode == "RGB" else image.convert("RGB")
    temporary = path.with_suffix(path.suffix + ".tmp")
    source.save(temporary, format=target_format)
    os.replace(temporary, path)


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.replace(temporary, path)


def _install_staged_export(
    staging_root: Path,
    output_root: Path,
    filenames: list[str],
    *,
    overwrite: bool,
) -> None:
    annotation = output_root / "incoming" / "winoground.jsonl"
    conflicts = [annotation] if annotation.exists() else []
    conflicts.extend(path for name in filenames if (path := output_root / "images" / name).exists())
    if conflicts and not overwrite:
        preview = ", ".join(path.as_posix() for path in conflicts[:3])
        raise FileExistsError(f"export targets already exist; use --overwrite: {preview}")

    image_root = output_root / "images"
    incoming_root = output_root / "incoming"
    image_root.mkdir(parents=True, exist_ok=True)
    incoming_root.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        os.replace(staging_root / "images" / name, image_root / name)
    os.replace(staging_root / "incoming" / "winoground.jsonl", annotation)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matches = sum(
        caption_multiset_matches(
            row["caption_0"],
            row["caption_1"],
            method=WINOGROUND_CONTENT_MULTISET,
        )
        for row in rows
    )
    return {
        "dataset_id": rows[0]["metadata"]["source_dataset"],
        "split": rows[0]["metadata"]["source_split"],
        "samples": len(rows),
        "unique_sample_ids": len({row["sample_id"] for row in rows}),
        "images": 2 * len(rows),
        "token_multiset_matches": matches,
        "token_multiset_mismatches": len(rows) - matches,
        "token_multiset_match_rate": 100.0 * matches / len(rows),
        "token_multiset_method": WINOGROUND_CONTENT_MULTISET,
    }


def main() -> int:
    args = parse_args()
    try:
        dataset = load_official_split(args.dataset_id, args.split)
        print(f"features: {dataset.features}")
        summary = export_winoground(
            dataset,
            args.output_root,
            dataset_id=args.dataset_id,
            split=args.split,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            max_samples=args.max_samples,
            image_format=args.image_format,
        )
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.max_samples is not None:
        print("warning: --max-samples is for testing/debugging and is not a formal export")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
