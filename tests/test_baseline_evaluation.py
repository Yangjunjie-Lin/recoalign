from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from recoalign.evaluation.baseline import evaluate_baseline, write_baseline_outputs


class FakeEncoder:
    def __init__(self) -> None:
        self.image_calls = 0
        self.text_calls = 0

    @property
    def fingerprint(self) -> dict[str, Any]:
        return {"framework": "fake", "model": "identity", "version": 1}

    def encode_image_paths(self, paths: list[Path], *, batch_size: int) -> np.ndarray:
        self.image_calls += 1
        vectors = []
        for path in paths:
            vectors.append([1.0, 0.0] if path.stem.startswith("a") else [0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)

    def encode_texts(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        self.text_calls += 1
        vectors = []
        for text in texts:
            vectors.append([1.0, 0.0] if text.startswith("a") else [0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


def test_flickr_baseline_and_embedding_cache(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (2, 2)).save(image_root / name)
    annotation = tmp_path / "test.jsonl"
    _write_jsonl(
        annotation,
        [
            {"image_id": "a", "image": "a.jpg", "captions": ["a one", "a two"]},
            {"image_id": "b", "image": "b.jpg", "captions": ["b one", "b two"]},
        ],
    )
    manifest = tmp_path / "dataset.yaml"
    manifest.write_text("manifest\n", encoding="utf-8")
    config = _config(tmp_path, "flickr30k", annotation, image_root, manifest)
    encoder = FakeEncoder()

    first = evaluate_baseline(config, encoder=encoder, project_root=tmp_path)
    assert first.metrics["i2t_R@1"] == 100.0
    assert first.metrics["t2i_R@1"] == 100.0
    assert first.metrics["mean_recall"] == 100.0
    assert first.metadata["cache"] == {
        "enabled": True,
        "images_hit": False,
        "texts_hit": False,
    }

    second = evaluate_baseline(config, encoder=encoder, project_root=tmp_path)
    assert second.metadata["cache"] == {
        "enabled": True,
        "images_hit": True,
        "texts_hit": True,
    }
    assert encoder.image_calls == 1
    assert encoder.text_calls == 1

    no_cache = evaluate_baseline(
        config, encoder=encoder, project_root=tmp_path, use_cache=False
    )
    assert no_cache.metadata["cache"] == {
        "enabled": False,
        "images_hit": False,
        "texts_hit": False,
    }
    assert encoder.image_calls == 2
    assert encoder.text_calls == 2


def test_sugarcrepe_reports_macro_and_category_accuracy(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (2, 2)).save(image_root / "a.jpg")
    Image.new("RGB", (2, 2)).save(image_root / "b.jpg")
    annotation = tmp_path / "test.jsonl"
    _write_jsonl(
        annotation,
        [
            {
                "sample_id": "replace_obj:0",
                "image": "a.jpg",
                "positive_caption": "a positive",
                "negative_caption": "b negative",
                "category": "replace_obj",
            },
            {
                "sample_id": "swap_obj:0",
                "image": "b.jpg",
                "positive_caption": "b positive",
                "negative_caption": "a negative",
                "category": "swap_obj",
            },
        ],
    )
    manifest = tmp_path / "dataset.yaml"
    manifest.write_text("manifest\n", encoding="utf-8")
    config = _config(tmp_path, "sugarcrepe", annotation, image_root, manifest)

    result = evaluate_baseline(config, encoder=FakeEncoder(), project_root=tmp_path)
    assert result.metrics["accuracy"] == 100.0
    assert result.metrics["macro_accuracy"] == 100.0
    assert result.metrics["accuracy/replace_obj"] == 100.0
    assert result.metrics["accuracy/swap_obj"] == 100.0
    assert result.metrics["tie_rate"] == 0.0
    assert len(result.predictions) == 2


def test_baseline_outputs_are_schema_validated(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (2, 2)).save(image_root / "a.jpg")
    annotation = tmp_path / "test.jsonl"
    _write_jsonl(
        annotation,
        [{"image_id": "a", "image": "a.jpg", "captions": ["a caption"]}],
    )
    manifest = tmp_path / "dataset.yaml"
    manifest.write_text("manifest\n", encoding="utf-8")
    result = evaluate_baseline(
        _config(tmp_path, "flickr30k", annotation, image_root, manifest),
        encoder=FakeEncoder(),
        project_root=tmp_path,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_baseline_outputs(run_dir, result, save_predictions=True)
    assert (run_dir / "evaluation.json").is_file()
    assert len((run_dir / "predictions.jsonl").read_text().splitlines()) == 2


def _config(
    tmp_path: Path,
    dataset: str,
    annotation: Path,
    image_root: Path,
    manifest: Path,
) -> dict[str, Any]:
    return {
        "experiment": {"name": "baseline", "seed": 42, "output_dir": str(tmp_path / "out")},
        "model": {
            "framework": "synthetic",
            "name": "identity",
            "pretrained": "fixture",
            "manifest": str(manifest),
            "precision": "fp32",
        },
        "data": {
            "dataset": dataset,
            "root": str(tmp_path),
            "manifest": str(manifest),
            "annotation_file": str(annotation),
            "image_root": str(image_root),
            "split": "test",
        },
        "evaluation": {
            "protocol": "synthetic-test",
            "image_batch_size": 2,
            "text_batch_size": 2,
            "ranking_batch_size": 1,
            "recall_at": [1, 2],
            "normalize_embeddings": True,
            "cache_dir": str(tmp_path / "cache"),
            "save_predictions": True,
        },
        "training": {"enabled": False},
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")
