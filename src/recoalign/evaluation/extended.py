"""ARO, Winoground, and BiVLC zero-shot evaluation implementations."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from recoalign.benchmarks.records import (
    load_multichoice_jsonl,
    load_paired_matrix_jsonl,
    require_images,
)
from recoalign.evaluation.cache import EmbeddingCache
from recoalign.evaluation.diagnostics import (
    evaluate_multichoice_scores,
    evaluate_paired_matrix_scores,
    summarize_multichoice,
    summarize_paired_matrix,
    token_multiset_match_rate,
)
from recoalign.models.base import VisionLanguageEncoder


@dataclass(frozen=True)
class ExtendedBenchmarkEvaluation:
    """Aggregate output returned to the main baseline runner."""

    metrics: dict[str, float]
    metadata: dict[str, Any]
    predictions: list[dict[str, Any]]


def evaluate_extended_benchmark(
    dataset: str,
    annotation_file: Path,
    image_root: Path,
    encoder: VisionLanguageEncoder,
    cache: EmbeddingCache,
    evaluation: dict[str, Any],
    common: dict[str, Any],
    *,
    use_cache: bool,
) -> ExtendedBenchmarkEvaluation:
    """Dispatch an extended compositional benchmark by normalized dataset name."""
    if dataset == "aro":
        return _evaluate_aro(
            annotation_file,
            image_root,
            encoder,
            cache,
            evaluation,
            common,
            use_cache=use_cache,
        )
    if dataset in {"winoground", "bivlc"}:
        return _evaluate_paired_matrix(
            dataset,
            annotation_file,
            image_root,
            encoder,
            cache,
            evaluation,
            common,
            use_cache=use_cache,
        )
    raise ValueError(f"unsupported extended benchmark: {dataset!r}")


def _evaluate_aro(
    annotation_file: Path,
    image_root: Path,
    encoder: VisionLanguageEncoder,
    cache: EmbeddingCache,
    evaluation: dict[str, Any],
    common: dict[str, Any],
    *,
    use_cache: bool,
) -> ExtendedBenchmarkEvaluation:
    records = load_multichoice_jsonl(annotation_file, image_root)
    require_images([record.image_path for record in records])
    _require_expected_count(evaluation, "expected_num_samples", len(records))

    subsets = [record.subset for record in records]
    _require_exact_values(evaluation, "required_subsets", subsets, label="ARO subset")

    image_keys = list(dict.fromkeys(str(record.image_path) for record in records))
    text_values = list(
        dict.fromkeys(caption for record in records for caption in record.captions)
    )
    image_index = {value: index for index, value in enumerate(image_keys)}
    text_index = {value: index for index, value in enumerate(text_values)}
    cache_metadata = {**common, "kind": "aro_multichoice"}

    image_started = time.perf_counter()
    image_cache = cache.get_or_compute(
        namespace="images",
        metadata=cache_metadata,
        identifiers=image_keys,
        compute=lambda: encoder.encode_image_paths(
            [Path(value) for value in image_keys],
            batch_size=int(evaluation["image_batch_size"]),
        ),
        enabled=use_cache,
    )
    image_seconds = time.perf_counter() - image_started

    text_started = time.perf_counter()
    text_cache = cache.get_or_compute(
        namespace="texts",
        metadata=cache_metadata,
        identifiers=text_values,
        compute=lambda: encoder.encode_texts(
            text_values,
            batch_size=int(evaluation["text_batch_size"]),
        ),
        enabled=use_cache,
    )
    text_seconds = time.perf_counter() - text_started

    image_norm_error = _max_norm_error(image_cache.embeddings)
    text_norm_error = _max_norm_error(text_cache.embeddings)

    scoring_started = time.perf_counter()
    score_rows: list[np.ndarray] = []
    correct_indices: list[int] = []
    caption_lists: list[tuple[str, ...]] = []
    for record in records:
        image_embedding = image_cache.embeddings[image_index[str(record.image_path)]]
        caption_embeddings = np.stack(
            [text_cache.embeddings[text_index[caption]] for caption in record.captions]
        )
        score_rows.append(caption_embeddings @ image_embedding)
        correct_indices.append(record.correct_index)
        caption_lists.append(record.captions)

    result = evaluate_multichoice_scores(score_rows, correct_indices)
    metrics = summarize_multichoice(result, subsets, caption_lists, correct_indices)
    scoring_seconds = time.perf_counter() - scoring_started

    predictions = [
        {
            "sample_id": record.sample_id,
            "subset": record.subset,
            "correct_index": int(record.correct_index),
            "predicted_index": int(result.predicted_indices[index]),
            "correct_score": float(result.correct_scores[index]),
            "top_score": float(result.top_scores[index]),
            "margin": float(result.margins[index]),
            "correct": bool(result.correct[index]),
            "tie": bool(result.ties[index]),
        }
        for index, record in enumerate(records)
    ]

    metadata = {
        **common,
        "benchmark": "aro_multichoice",
        "num_samples": len(records),
        "num_unique_images": len(image_keys),
        "num_unique_captions": len(text_values),
        "subset_counts": dict(sorted(Counter(subsets).items())),
        "choice_count_min": min(len(record.captions) for record in records),
        "choice_count_max": max(len(record.captions) for record in records),
        "embedding_dimension": int(image_cache.embeddings.shape[1]),
        "max_image_norm_error": image_norm_error,
        "max_text_norm_error": text_norm_error,
        "cache": {"images_hit": image_cache.hit, "texts_hit": text_cache.hit},
        "timing_seconds": {
            "image_stage": image_seconds,
            "text_stage": text_seconds,
            "scoring": scoring_seconds,
        },
        "tie_policy": "correct score must be strictly greater than every distractor",
        "blind_controls": [
            "majority correct-index prior",
            "shortest-caption heuristic",
            "longest-caption heuristic",
        ],
    }
    return ExtendedBenchmarkEvaluation(metrics, metadata, predictions)


def _evaluate_paired_matrix(
    dataset: str,
    annotation_file: Path,
    image_root: Path,
    encoder: VisionLanguageEncoder,
    cache: EmbeddingCache,
    evaluation: dict[str, Any],
    common: dict[str, Any],
    *,
    use_cache: bool,
) -> ExtendedBenchmarkEvaluation:
    records = load_paired_matrix_jsonl(annotation_file, image_root)
    require_images(
        [path for record in records for path in (record.image_0_path, record.image_1_path)]
    )
    _require_expected_count(evaluation, "expected_num_samples", len(records))

    categories = [record.category for record in records]
    _require_exact_values(
        evaluation,
        "required_categories",
        categories,
        label=f"{dataset} category",
    )

    image_keys = list(
        dict.fromkeys(
            str(path)
            for record in records
            for path in (record.image_0_path, record.image_1_path)
        )
    )
    text_values = list(
        dict.fromkeys(
            caption
            for record in records
            for caption in (record.caption_0, record.caption_1)
        )
    )
    image_index = {value: index for index, value in enumerate(image_keys)}
    text_index = {value: index for index, value in enumerate(text_values)}
    cache_metadata = {**common, "kind": f"{dataset}_paired_matrix"}

    image_started = time.perf_counter()
    image_cache = cache.get_or_compute(
        namespace="images",
        metadata=cache_metadata,
        identifiers=image_keys,
        compute=lambda: encoder.encode_image_paths(
            [Path(value) for value in image_keys],
            batch_size=int(evaluation["image_batch_size"]),
        ),
        enabled=use_cache,
    )
    image_seconds = time.perf_counter() - image_started

    text_started = time.perf_counter()
    text_cache = cache.get_or_compute(
        namespace="texts",
        metadata=cache_metadata,
        identifiers=text_values,
        compute=lambda: encoder.encode_texts(
            text_values,
            batch_size=int(evaluation["text_batch_size"]),
        ),
        enabled=use_cache,
    )
    text_seconds = time.perf_counter() - text_started

    image_norm_error = _max_norm_error(image_cache.embeddings)
    text_norm_error = _max_norm_error(text_cache.embeddings)

    scoring_started = time.perf_counter()
    scores = np.empty((len(records), 2, 2), dtype=np.float32)
    for index, record in enumerate(records):
        image_0 = image_cache.embeddings[image_index[str(record.image_0_path)]]
        image_1 = image_cache.embeddings[image_index[str(record.image_1_path)]]
        caption_0 = text_cache.embeddings[text_index[record.caption_0]]
        caption_1 = text_cache.embeddings[text_index[record.caption_1]]
        scores[index] = np.asarray(
            [
                [float(image_0 @ caption_0), float(image_0 @ caption_1)],
                [float(image_1 @ caption_0), float(image_1 @ caption_1)],
            ],
            dtype=np.float32,
        )

    result = evaluate_paired_matrix_scores(scores)
    metrics = summarize_paired_matrix(
        result,
        categories,
        tags=[record.tags for record in records],
    )
    multiset_rate = token_multiset_match_rate(
        [(record.caption_0, record.caption_1) for record in records]
    )
    metrics["caption_token_multiset_match_rate"] = multiset_rate
    if evaluation.get("require_caption_token_multiset_match", False) and multiset_rate != 100.0:
        raise ValueError(
            f"{dataset} requires identical caption token multisets; observed {multiset_rate:.4f}%"
        )
    scoring_seconds = time.perf_counter() - scoring_started

    predictions = [
        {
            "sample_id": record.sample_id,
            "category": record.category,
            "tags": list(record.tags),
            "scores": [float(value) for value in result.scores[index].reshape(-1)],
            "image_to_text_correct": bool(result.image_to_text_correct[index]),
            "text_to_image_correct": bool(result.text_to_image_correct[index]),
            "group_correct": bool(result.group_correct[index]),
            "tie": bool(result.ties[index]),
        }
        for index, record in enumerate(records)
    ]

    tag_counts = Counter(tag for record in records for tag in record.tags)
    metadata = {
        **common,
        "benchmark": f"{dataset}_paired_matrix",
        "num_samples": len(records),
        "num_unique_images": len(image_keys),
        "num_unique_captions": len(text_values),
        "category_counts": dict(sorted(Counter(categories).items())),
        "tag_counts": dict(sorted(tag_counts.items())),
        "embedding_dimension": int(image_cache.embeddings.shape[1]),
        "max_image_norm_error": image_norm_error,
        "max_text_norm_error": text_norm_error,
        "caption_token_multiset_match_rate": multiset_rate,
        "cache": {"images_hit": image_cache.hit, "texts_hit": text_cache.hit},
        "timing_seconds": {
            "image_stage": image_seconds,
            "text_stage": text_seconds,
            "scoring": scoring_seconds,
        },
        "score_matrix_convention": "scores[image_index, caption_index], diagonal is correct",
        "tie_policy": "all four diagonal-vs-off-diagonal comparisons are strict",
    }
    return ExtendedBenchmarkEvaluation(metrics, metadata, predictions)


def _require_expected_count(evaluation: dict[str, Any], key: str, observed: int) -> None:
    expected = evaluation.get(key)
    if expected is None:
        return
    if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
        raise ValueError(f"evaluation.{key} must be a positive integer")
    if observed != expected:
        raise ValueError(f"evaluation.{key} expected {expected}, observed {observed}")


def _require_exact_values(
    evaluation: dict[str, Any],
    key: str,
    observed_values: list[str],
    *,
    label: str,
) -> None:
    required = evaluation.get(key)
    if required is None:
        return
    expected = set(required)
    observed = set(observed_values)
    if observed != expected:
        raise ValueError(
            f"{label} mismatch: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _max_norm_error(embeddings: np.ndarray, *, tolerance: float = 1e-3) -> float:
    norms = np.linalg.norm(np.asarray(embeddings, dtype=np.float32), axis=1)
    error = float(np.max(np.abs(norms - 1.0)))
    if error > tolerance:
        raise ValueError(
            f"encoder embeddings must be L2-normalized; maximum norm error is {error:.6f}"
        )
    return error
