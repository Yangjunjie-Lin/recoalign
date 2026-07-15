"""Reproducible zero-shot baseline evaluation for retrieval and compositionality."""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from recoalign.benchmarks.records import load_pairwise_jsonl, load_retrieval_jsonl, require_images
from recoalign.config import file_digest
from recoalign.evaluation.cache import EmbeddingCache
from recoalign.evaluation.compositional import evaluate_pairwise_scores, summarize_pairwise
from recoalign.evaluation.retrieval import rank_queries, summarize_ranks
from recoalign.models.base import VisionLanguageEncoder
from recoalign.models.openclip_encoder import OpenCLIPConfig, OpenCLIPEncoder
from recoalign.reproducibility import atomic_write_json, seed_everything, utc_now
from recoalign.schema_validation import repository_root, validate_payload


@dataclass(frozen=True)
class BaselineEvaluation:
    metrics: dict[str, float]
    metadata: dict[str, Any]
    predictions: list[dict[str, Any]]


def evaluate_baseline(
    config: dict[str, Any],
    *,
    encoder: VisionLanguageEncoder | None = None,
    use_cache: bool = True,
    project_root: str | Path | None = None,
) -> BaselineEvaluation:
    """Dispatch one experiment config to a trusted benchmark implementation."""
    _validate_baseline_config(config, require_openclip=encoder is None)
    root = Path(project_root) if project_root is not None else repository_root()
    seed_everything(int(config["experiment"]["seed"]), deterministic=True)
    model = encoder or _openclip_from_config(config)
    data = config["data"]
    evaluation = config["evaluation"]
    dataset = str(data["dataset"]).lower()
    annotation_file = _resolve(data["annotation_file"], root)
    image_root = _resolve(data["image_root"], root)
    cache = EmbeddingCache(_resolve(evaluation.get("cache_dir", "data/cache/embeddings"), root))
    common = {
        "protocol_version": 1,
        "protocol": evaluation["protocol"],
        "dataset": dataset,
        "split": data["split"],
        "annotation_sha256": file_digest(annotation_file),
        "dataset_manifest_sha256": file_digest(_resolve(data["manifest"], root)),
        "encoder": model.fingerprint,
    }
    started = time.perf_counter()
    if dataset in {"flickr30k", "mscoco"}:
        result = _evaluate_retrieval(
            annotation_file, image_root, model, cache, evaluation, common, use_cache=use_cache
        )
    elif dataset == "sugarcrepe":
        result = _evaluate_sugarcrepe(
            annotation_file, image_root, model, cache, evaluation, common, use_cache=use_cache
        )
    else:
        raise ValueError(f"unsupported baseline dataset: {data['dataset']!r}")
    return _with_runtime_metadata(result, time.perf_counter() - started, model)


def write_baseline_outputs(
    run_dir: str | Path,
    result: BaselineEvaluation,
    *,
    save_predictions: bool,
) -> None:
    """Write aggregate metrics and auditable per-query predictions."""
    directory = Path(run_dir)
    atomic_write_json(directory / "metrics.pending.json", result.metrics)
    predictions_file: str | None = None
    if save_predictions:
        predictions_path = directory / "predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in result.predictions:
                validate_payload("prediction", row)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        predictions_file = predictions_path.name
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "metrics": result.metrics,
        "metadata": result.metadata,
        "predictions_file": predictions_file,
    }
    validate_payload("evaluation", payload)
    atomic_write_json(directory / "evaluation.json", payload)


def _evaluate_retrieval(
    annotation_file: Path,
    image_root: Path,
    encoder: VisionLanguageEncoder,
    cache: EmbeddingCache,
    evaluation: dict[str, Any],
    common: dict[str, Any],
    *,
    use_cache: bool,
) -> BaselineEvaluation:
    records = load_retrieval_jsonl(annotation_file, image_root)
    image_paths = [record.image_path for record in records]
    require_images(image_paths)
    image_ids = [record.image_id for record in records]

    captions: list[str] = []
    caption_ids: list[str] = []
    image_positives: list[list[int]] = []
    caption_positives: list[list[int]] = []
    for image_index, record in enumerate(records):
        positives = []
        for caption_index, caption in enumerate(record.captions):
            global_index = len(captions)
            captions.append(caption)
            caption_ids.append(f"{record.image_id}#{caption_index}")
            positives.append(global_index)
            caption_positives.append([image_index])
        image_positives.append(positives)

    _require_expected_count(evaluation, "expected_num_images", len(records))
    _require_expected_count(evaluation, "expected_num_captions", len(captions))
    metadata_key = {**common, "kind": "standard_retrieval"}
    image_started = time.perf_counter()
    image_cache = cache.get_or_compute(
        namespace="images",
        metadata=metadata_key,
        identifiers=image_ids,
        compute=lambda: encoder.encode_image_paths(
            image_paths, batch_size=int(evaluation["image_batch_size"])
        ),
        enabled=use_cache,
    )
    image_seconds = time.perf_counter() - image_started
    text_started = time.perf_counter()
    text_cache = cache.get_or_compute(
        namespace="texts",
        metadata=metadata_key,
        identifiers=caption_ids,
        compute=lambda: encoder.encode_texts(
            captions, batch_size=int(evaluation["text_batch_size"])
        ),
        enabled=use_cache,
    )
    text_seconds = time.perf_counter() - text_started
    image_norm_error = _max_norm_error(image_cache.embeddings)
    text_norm_error = _max_norm_error(text_cache.embeddings)

    ranking_batch_size = int(evaluation.get("ranking_batch_size", 128))
    ranking_started = time.perf_counter()
    i2t = rank_queries(
        image_cache.embeddings,
        text_cache.embeddings,
        image_positives,
        batch_size=ranking_batch_size,
    )
    t2i = rank_queries(
        text_cache.embeddings,
        image_cache.embeddings,
        caption_positives,
        batch_size=ranking_batch_size,
    )
    metrics = summarize_ranks(i2t, t2i, evaluation["recall_at"])
    ranking_seconds = time.perf_counter() - ranking_started

    predictions: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        predictions.append(
            _rank_prediction("image_to_text", record.image_id, i2t, index, caption_ids)
        )
    for index, caption_id in enumerate(caption_ids):
        predictions.append(_rank_prediction("text_to_image", caption_id, t2i, index, image_ids))

    metadata = {
        **common,
        "benchmark": f"{common['dataset']}_standard_retrieval",
        "num_images": len(records),
        "num_captions": len(captions),
        "captions_per_image_min": min(len(record.captions) for record in records),
        "captions_per_image_max": max(len(record.captions) for record in records),
        "ranking_batch_size": ranking_batch_size,
        "embedding_dimension": int(image_cache.embeddings.shape[1]),
        "max_image_norm_error": image_norm_error,
        "max_text_norm_error": text_norm_error,
        "cache": {"images_hit": image_cache.hit, "texts_hit": text_cache.hit},
        "timing_seconds": {
            "image_stage": image_seconds,
            "text_stage": text_seconds,
            "ranking": ranking_seconds,
        },
        "tie_policy": "stable descending score, then ascending candidate index",
    }
    return BaselineEvaluation(metrics, metadata, predictions)


def _evaluate_sugarcrepe(
    annotation_file: Path,
    image_root: Path,
    encoder: VisionLanguageEncoder,
    cache: EmbeddingCache,
    evaluation: dict[str, Any],
    common: dict[str, Any],
    *,
    use_cache: bool,
) -> BaselineEvaluation:
    records = load_pairwise_jsonl(annotation_file, image_root)
    require_images([record.image_path for record in records])
    image_keys = list(dict.fromkeys(str(record.image_path) for record in records))
    text_values = list(
        dict.fromkeys(
            caption
            for record in records
            for caption in (record.positive_caption, record.negative_caption)
        )
    )
    image_index = {value: index for index, value in enumerate(image_keys)}
    text_index = {value: index for index, value in enumerate(text_values)}
    metadata_key = {**common, "kind": "pairwise_compositional"}

    image_started = time.perf_counter()
    image_cache = cache.get_or_compute(
        namespace="images",
        metadata=metadata_key,
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
        metadata=metadata_key,
        identifiers=text_values,
        compute=lambda: encoder.encode_texts(
            text_values, batch_size=int(evaluation["text_batch_size"])
        ),
        enabled=use_cache,
    )
    text_seconds = time.perf_counter() - text_started
    image_norm_error = _max_norm_error(image_cache.embeddings)
    text_norm_error = _max_norm_error(text_cache.embeddings)

    scoring_started = time.perf_counter()
    positive_scores = np.empty(len(records), dtype=np.float32)
    negative_scores = np.empty(len(records), dtype=np.float32)
    for index, record in enumerate(records):
        image_embedding = image_cache.embeddings[image_index[str(record.image_path)]]
        positive_scores[index] = float(
            image_embedding @ text_cache.embeddings[text_index[record.positive_caption]]
        )
        negative_scores[index] = float(
            image_embedding @ text_cache.embeddings[text_index[record.negative_caption]]
        )
    pairwise = evaluate_pairwise_scores(positive_scores, negative_scores)
    scoring_seconds = time.perf_counter() - scoring_started
    categories = [record.category for record in records]
    required_categories = evaluation.get("required_categories")
    if required_categories is not None:
        expected, observed = set(required_categories), set(categories)
        if observed != expected:
            raise ValueError(
                "SugarCrepe category mismatch: "
                f"missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}"
            )
    _require_expected_count(evaluation, "expected_num_samples", len(records))
    metrics = summarize_pairwise(pairwise, categories)
    predictions = [
        {
            "sample_id": record.sample_id,
            "category": record.category,
            "positive_score": float(pairwise.positive_scores[index]),
            "negative_score": float(pairwise.negative_scores[index]),
            "correct": bool(pairwise.correct[index]),
            "tie": bool(pairwise.ties[index]),
        }
        for index, record in enumerate(records)
    ]
    metadata = {
        **common,
        "benchmark": "sugarcrepe_pairwise",
        "num_samples": len(records),
        "num_unique_images": len(image_keys),
        "category_counts": dict(sorted(Counter(categories).items())),
        "embedding_dimension": int(image_cache.embeddings.shape[1]),
        "max_image_norm_error": image_norm_error,
        "max_text_norm_error": text_norm_error,
        "cache": {"images_hit": image_cache.hit, "texts_hit": text_cache.hit},
        "timing_seconds": {
            "image_stage": image_seconds,
            "text_stage": text_seconds,
            "scoring": scoring_seconds,
        },
        "tie_policy": "strict positive_score > negative_score; ties are incorrect and reported",
    }
    return BaselineEvaluation(metrics, metadata, predictions)


def _rank_prediction(
    direction: str,
    query_id: str,
    result: Any,
    index: int,
    candidate_ids: list[str],
) -> dict[str, Any]:
    return {
        "direction": direction,
        "query_id": query_id,
        "best_positive_rank": int(result.ranks[index]),
        "best_positive_candidate_id": candidate_ids[int(result.best_positive_indices[index])],
        "best_positive_score": float(result.best_positive_scores[index]),
        "top1_candidate_id": candidate_ids[int(result.top1_indices[index])],
        "top1_score": float(result.top1_scores[index]),
    }


def _max_norm_error(embeddings: np.ndarray, *, tolerance: float = 1e-3) -> float:
    norms = np.linalg.norm(np.asarray(embeddings, dtype=np.float32), axis=1)
    error = float(np.max(np.abs(norms - 1.0)))
    if error > tolerance:
        raise ValueError(
            f"encoder embeddings must be L2-normalized; maximum norm error is {error:.6f}"
        )
    return error


def _with_runtime_metadata(
    result: BaselineEvaluation,
    seconds: float,
    encoder: VisionLanguageEncoder,
) -> BaselineEvaluation:
    metadata = dict(result.metadata)
    timing = dict(metadata.get("timing_seconds", {}))
    timing["total"] = float(seconds)
    metadata["timing_seconds"] = timing
    runtime = getattr(encoder, "runtime_metadata", None)
    if isinstance(runtime, dict):
        metadata["encoder_runtime"] = runtime
    return BaselineEvaluation(result.metrics, metadata, result.predictions)


def _require_expected_count(evaluation: dict[str, Any], key: str, observed: int) -> None:
    expected = evaluation.get(key)
    if expected is None:
        return
    if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
        raise ValueError(f"evaluation.{key} must be a positive integer")
    if observed != expected:
        raise ValueError(f"evaluation.{key} expected {expected}, observed {observed}")


def _validate_baseline_config(config: dict[str, Any], *, require_openclip: bool) -> None:
    required = (
        "data.annotation_file",
        "data.image_root",
        "evaluation.image_batch_size",
        "evaluation.text_batch_size",
    )
    for dotted in required:
        value: Any = config
        for part in dotted.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ValueError(f"baseline configuration is missing {dotted}")
            value = value[part]
    for key in ("image_batch_size", "text_batch_size", "ranking_batch_size"):
        value = config["evaluation"].get(key, 128 if key == "ranking_batch_size" else None)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"evaluation.{key} must be a positive integer")
    protocol = config["evaluation"].get("protocol")
    if not isinstance(protocol, str) or not protocol.strip():
        raise ValueError("evaluation.protocol must be a non-empty string")
    for key in ("expected_num_images", "expected_num_captions", "expected_num_samples"):
        value = config["evaluation"].get(key)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ):
            raise ValueError(f"evaluation.{key} must be a positive integer")
    categories = config["evaluation"].get("required_categories")
    if categories is not None and (
        not isinstance(categories, list)
        or not categories
        or any(not isinstance(item, str) or not item.strip() for item in categories)
        or len(set(categories)) != len(categories)
    ):
        raise ValueError("evaluation.required_categories must be unique non-empty strings")
    if config["evaluation"].get("normalize_embeddings", True) is not True:
        raise ValueError("credible baseline evaluation requires normalized embeddings")
    if not isinstance(config["evaluation"].get("save_predictions", True), bool):
        raise ValueError("evaluation.save_predictions must be a boolean")
    if str(config["data"]["dataset"]).lower() not in {"flickr30k", "mscoco", "sugarcrepe"}:
        raise ValueError(f"unsupported baseline dataset: {config['data']['dataset']!r}")
    if require_openclip and config["model"]["framework"] != "open_clip":
        raise ValueError("run-baseline currently requires model.framework=open_clip")


def _openclip_from_config(config: dict[str, Any]) -> OpenCLIPEncoder:
    model = config["model"]
    return OpenCLIPEncoder(
        OpenCLIPConfig(
            model_name=model["name"],
            pretrained=model["pretrained"],
            device=model.get("device", "cuda"),
            precision=model.get("precision", "amp"),
        )
    )


def _resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path
