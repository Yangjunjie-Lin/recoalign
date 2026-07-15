"""Validated records shared by retrieval and pairwise compositional benchmarks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RetrievalRecord:
    """One image and all captions considered positive for that image."""

    image_id: str
    image_path: Path
    captions: tuple[str, ...]


@dataclass(frozen=True)
class PairwiseCaptionRecord:
    """One image with one positive and one compositional hard-negative caption."""

    sample_id: str
    image_path: Path
    positive_caption: str
    negative_caption: str
    category: str


def load_retrieval_jsonl(
    annotation_file: str | Path,
    image_root: str | Path,
) -> list[RetrievalRecord]:
    """Load the normalized one-image-per-row retrieval format."""
    rows = _load_jsonl(annotation_file)
    root = Path(image_root)
    records: list[RetrievalRecord] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        image_id = _required_string(row, "image_id", index)
        image = _required_string(row, "image", index)
        captions = row.get("captions")
        if not isinstance(captions, list) or not captions:
            raise ValueError(f"row {index}: captions must be a non-empty list")
        normalized_captions = tuple(
            _nonempty_text(value, f"row {index}: caption") for value in captions
        )
        if image_id in seen:
            raise ValueError(f"row {index}: duplicate image_id {image_id!r}")
        seen.add(image_id)
        records.append(
            RetrievalRecord(
                image_id=image_id,
                image_path=root / image,
                captions=normalized_captions,
            )
        )
    if not records:
        raise ValueError("retrieval annotations contain no records")
    return records


def load_pairwise_jsonl(
    annotation_file: str | Path,
    image_root: str | Path,
) -> list[PairwiseCaptionRecord]:
    """Load the normalized SugarCrepe-style pairwise format."""
    rows = _load_jsonl(annotation_file)
    root = Path(image_root)
    records: list[PairwiseCaptionRecord] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        sample_id = _required_string(row, "sample_id", index)
        if sample_id in seen:
            raise ValueError(f"row {index}: duplicate sample_id {sample_id!r}")
        seen.add(sample_id)
        records.append(
            PairwiseCaptionRecord(
                sample_id=sample_id,
                image_path=root / _required_string(row, "image", index),
                positive_caption=_required_string(row, "positive_caption", index),
                negative_caption=_required_string(row, "negative_caption", index),
                category=_required_string(row, "category", index),
            )
        )
    if not records:
        raise ValueError("pairwise annotations contain no records")
    return records


def require_images(paths: list[Path]) -> None:
    """Fail early with a compact message when referenced images are unavailable."""
    missing = [path for path in paths if not path.is_file()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        suffix = "" if len(missing) <= 3 else f" (+{len(missing) - 3} more)"
        raise FileNotFoundError(f"missing benchmark images: {preview}{suffix}")


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"annotation file does not exist: {source}")
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {source}:{line_number}: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"row {line_number}: JSONL value must be an object")
            rows.append(payload)
    return rows


def _required_string(row: dict[str, Any], key: str, index: int) -> str:
    if key not in row:
        raise ValueError(f"row {index}: missing field {key!r}")
    return _nonempty_text(row[key], f"row {index}: {key}")


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()
