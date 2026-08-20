"""Benchmark-neutral records and normalized dataset adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from recoalign.benchmarks.records import (
    load_multichoice_jsonl,
    load_paired_matrix_jsonl,
    load_pairwise_jsonl,
    require_images,
)

SUPPORTED_BENCHMARKS = ("sugarcrepe", "aro", "winoground", "crepe", "gqa", "mmvp")
SUPPORTED_FORMATS = ("pairwise", "multichoice", "paired_matrix", "qa")


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    group_id: str
    image: Path
    question: str
    answer: str
    choices: tuple[str, ...]
    choice_texts: tuple[str, ...]
    category: str
    dimension: str
    tags: tuple[str, ...] = ()
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    category: str
    split: str
    format: str
    samples: tuple[BenchmarkSample, ...]
    manifest: Path
    annotation_file: Path
    aggregation: str = "accuracy"


def load_benchmark_dataset(
    config: dict[str, Any], *, project_root: str | Path | None = None
) -> BenchmarkDataset:
    benchmark = config.get("benchmark", config)
    if not isinstance(benchmark, dict):
        raise ValueError("benchmark configuration must be a mapping")
    name = str(benchmark.get("name", "")).lower()
    if name not in SUPPORTED_BENCHMARKS:
        raise ValueError(f"unsupported comprehensive benchmark: {name!r}")
    format_name = str(benchmark.get("format", "")).lower()
    if format_name not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported benchmark format: {format_name!r}")
    root = Path(project_root or Path.cwd()).resolve()
    annotation = _resolve(benchmark.get("annotation_file"), root)
    image_root = _resolve(benchmark.get("image_root"), root)
    manifest = _resolve(benchmark.get("manifest"), root)
    if not manifest.is_file():
        raise FileNotFoundError(f"benchmark manifest does not exist: {manifest}")
    if format_name == "pairwise":
        samples = _pairwise(annotation, image_root)
    elif format_name == "multichoice":
        samples = _multichoice(annotation, image_root)
    elif format_name == "paired_matrix":
        samples = _paired_matrix(annotation, image_root)
    else:
        samples = _qa(annotation, image_root)
    require_images([sample.image for sample in samples])
    expected = benchmark.get("expected_samples")
    observed_units = (
        len({sample.group_id for sample in samples})
        if format_name == "paired_matrix"
        else len(samples)
    )
    if expected is not None and int(expected) != observed_units:
        raise ValueError(
            f"{name} expected {int(expected)} evaluation units, observed {observed_units}"
        )
    return BenchmarkDataset(
        name=name,
        category=str(benchmark.get("category", "compositional")),
        split=str(benchmark.get("split", "test")),
        format=format_name,
        samples=tuple(samples),
        manifest=manifest,
        annotation_file=annotation,
        aggregation="paired_group" if format_name == "paired_matrix" else "accuracy",
    )


def inspect_benchmark_availability(
    config: dict[str, Any], *, project_root: str | Path | None = None
) -> dict[str, Any]:
    benchmark = config.get("benchmark", config)
    root = Path(project_root or Path.cwd()).resolve()
    paths = {
        key: _resolve(benchmark.get(key), root)
        for key in ("manifest", "annotation_file", "image_root")
    }
    if str(benchmark.get("format", "")).lower() == "registered_experiment":
        return {
            "benchmark": benchmark.get("name"),
            "format": "registered_experiment",
            "registered_experiment": benchmark.get("registered_experiment"),
            "paths": {key: path.as_posix() for key, path in paths.items()},
            "manifest_exists": paths["manifest"].is_file(),
            "annotation_exists": False,
            "image_root_exists": False,
            "available": True,
            "execution": "delegate_to_phase1_registered_runner",
        }
    return {
        "benchmark": benchmark.get("name"),
        "format": benchmark.get("format"),
        "paths": {key: path.as_posix() for key, path in paths.items()},
        "manifest_exists": paths["manifest"].is_file(),
        "annotation_exists": paths["annotation_file"].is_file(),
        "image_root_exists": paths["image_root"].is_dir(),
        "available": all(
            (
                paths["manifest"].is_file(),
                paths["annotation_file"].is_file(),
                paths["image_root"].is_dir(),
            )
        ),
    }


def _pairwise(annotation: Path, image_root: Path) -> list[BenchmarkSample]:
    rows = load_pairwise_jsonl(annotation, image_root)
    labels = ("A", "B")
    return [
        BenchmarkSample(
            sample_id=row.sample_id,
            group_id=row.sample_id,
            image=row.image_path,
            question="Which caption best describes this image?",
            answer="A",
            choices=labels,
            choice_texts=(row.positive_caption, row.negative_caption),
            category=row.category,
            dimension=_dimension(row.category),
        )
        for row in rows
    ]


def _multichoice(annotation: Path, image_root: Path) -> list[BenchmarkSample]:
    rows = load_multichoice_jsonl(annotation, image_root)
    samples: list[BenchmarkSample] = []
    for row in rows:
        labels = _labels(len(row.captions))
        samples.append(
            BenchmarkSample(
                sample_id=row.sample_id,
                group_id=row.sample_id,
                image=row.image_path,
                question="Which caption best describes this image?",
                answer=labels[row.correct_index],
                choices=labels,
                choice_texts=row.captions,
                category=row.subset,
                dimension=_dimension(row.subset, row.tags),
                tags=row.tags,
                context=_context(row.metadata),
                metadata=row.metadata,
            )
        )
    return samples


def _paired_matrix(annotation: Path, image_root: Path) -> list[BenchmarkSample]:
    rows = load_paired_matrix_jsonl(annotation, image_root)
    samples: list[BenchmarkSample] = []
    for row in rows:
        choice_texts = (row.caption_0, row.caption_1)
        for index, image in enumerate((row.image_0_path, row.image_1_path)):
            samples.append(
                BenchmarkSample(
                    sample_id=f"{row.sample_id}:image_{index}",
                    group_id=row.sample_id,
                    image=image,
                    question="Which caption best describes this image?",
                    answer=("A", "B")[index],
                    choices=("A", "B"),
                    choice_texts=choice_texts,
                    category=row.category,
                    dimension=_dimension(row.category, row.tags),
                    tags=row.tags,
                    context=_context(row.metadata),
                    metadata=row.metadata,
                )
            )
    return samples


def _qa(annotation: Path, image_root: Path) -> list[BenchmarkSample]:
    rows = _jsonl(annotation)
    samples: list[BenchmarkSample] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        sample_id = _text(row, "sample_id", index)
        if sample_id in seen:
            raise ValueError(f"row {index}: duplicate sample_id {sample_id!r}")
        seen.add(sample_id)
        choices_raw = row.get("choices", [])
        if not isinstance(choices_raw, list):
            raise ValueError(f"row {index}: choices must be a list")
        choice_texts = tuple(str(value).strip() for value in choices_raw if str(value).strip())
        labels = _labels(len(choice_texts)) if choice_texts else ()
        answer_index = row.get("answer_index")
        if answer_index is not None:
            if not isinstance(answer_index, int) or not 0 <= answer_index < len(labels):
                raise ValueError(f"row {index}: answer_index must reference choices")
            answer = labels[answer_index]
        else:
            answer = _text(row, "answer", index)
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"row {index}: metadata must be a mapping")
        context_payload = {
            **metadata,
            **{key: row.get(key) for key in ("caption", "oracle_graph")},
        }
        context = _context(context_payload)
        category = str(row.get("category", "general")).strip() or "general"
        tags_raw = row.get("tags", [])
        if not isinstance(tags_raw, list):
            raise ValueError(f"row {index}: tags must be a list")
        tags = tuple(str(value) for value in tags_raw)
        samples.append(
            BenchmarkSample(
                sample_id=sample_id,
                group_id=str(row.get("group_id", sample_id)),
                image=image_root / _text(row, "image", index),
                question=_text(row, "question", index),
                answer=answer,
                choices=labels,
                choice_texts=choice_texts,
                category=category,
                dimension=str(row.get("dimension", _dimension(category, tags))),
                tags=tags,
                context=context,
                metadata=metadata,
            )
        )
    if not samples:
        raise ValueError("QA annotations contain no records")
    return samples


def _context(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metadata[key]
        for key in ("caption", "oracle_graph")
        if metadata.get(key) not in (None, "", [], {})
    }


def _dimension(category: str, tags: tuple[str, ...] = ()) -> str:
    value = " ".join((category, *tags)).casefold()
    if any(token in value for token in ("attribute", "adjective", "color", "size", "att")):
        return "attribute"
    if any(token in value for token in ("relation", "preposition", "spatial", "rel")):
        return "relation"
    if any(token in value for token in ("object", "noun", "obj")):
        return "object"
    return "composition"


def _labels(count: int) -> tuple[str, ...]:
    if count > 26:
        raise ValueError("multiple-choice evaluation supports at most 26 candidates")
    return tuple(chr(ord("A") + index) for index in range(count))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"annotation file does not exist: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
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


def _text(row: dict[str, Any], key: str, index: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row {index}: {key} must be a non-empty string")
    return value.strip()


def _resolve(value: Any, root: Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("benchmark path fields must be non-empty")
    path = Path(value)
    return path if path.is_absolute() else root / path


__all__ = [
    "BenchmarkDataset",
    "BenchmarkSample",
    "SUPPORTED_BENCHMARKS",
    "inspect_benchmark_availability",
    "load_benchmark_dataset",
]
