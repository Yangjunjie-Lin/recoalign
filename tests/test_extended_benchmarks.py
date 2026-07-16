from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from recoalign.evaluation.baseline import evaluate_baseline, write_baseline_outputs


class DiagnosticEncoder:
    @property
    def fingerprint(self) -> dict[str, Any]:
        return {"framework": "fake", "model": "diagnostic", "version": 1}

    def encode_image_paths(self, paths: list[Path], *, batch_size: int) -> np.ndarray:
        return np.asarray(
            [[1.0, 0.0] if path.stem.startswith("a") else [0.0, 1.0] for path in paths],
            dtype=np.float32,
        )

    def encode_texts(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        return np.asarray(
            [[1.0, 0.0] if text.startswith("a") else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )


def test_aro_baseline_reports_subsets_and_blind_controls(tmp_path: Path) -> None:
    image_root = _images(tmp_path)
    annotation = tmp_path / "aro.jsonl"
    _write_jsonl(
        annotation,
        [
            {
                "sample_id": "relation:0",
                "image": "a.jpg",
                "captions": ["a correct", "b wrong"],
                "correct_index": 0,
                "subset": "vg_relation",
            },
            {
                "sample_id": "attribute:0",
                "image": "b.jpg",
                "captions": ["a wrong", "b correct"],
                "correct_index": 1,
                "subset": "vg_attribution",
            },
        ],
    )
    config = _config(tmp_path, "aro", annotation, image_root)
    config["evaluation"]["required_subsets"] = ["vg_relation", "vg_attribution"]

    result = evaluate_baseline(config, encoder=DiagnosticEncoder(), project_root=tmp_path)

    assert result.metrics["accuracy"] == 100.0
    assert result.metrics["macro_subset_accuracy"] == 100.0
    assert result.metrics["accuracy/vg_relation"] == 100.0
    assert result.metrics["accuracy/vg_attribution"] == 100.0
    assert "blind/majority_index_accuracy" in result.metrics
    assert len(result.predictions) == 2


def test_winoground_baseline_reports_directional_and_group_scores(tmp_path: Path) -> None:
    image_root = _images(tmp_path)
    annotation = tmp_path / "winoground.jsonl"
    _write_jsonl(
        annotation,
        [
            {
                "sample_id": "0",
                "image_0": "a.jpg",
                "image_1": "b.jpg",
                "caption_0": "a b",
                "caption_1": "b a",
                "category": "winoground",
                "tags": ["spatial"],
            }
        ],
    )
    config = _config(tmp_path, "winoground", annotation, image_root)
    config["evaluation"]["required_categories"] = ["winoground"]
    config["evaluation"]["require_caption_token_multiset_match"] = True

    result = evaluate_baseline(config, encoder=DiagnosticEncoder(), project_root=tmp_path)

    assert result.metrics["image_to_text_accuracy"] == 100.0
    assert result.metrics["text_to_image_accuracy"] == 100.0
    assert result.metrics["group_accuracy"] == 100.0
    assert result.metrics["caption_token_multiset_match_rate"] == 100.0
    assert result.metrics["group_accuracy/tag/spatial"] == 100.0

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_baseline_outputs(run_dir, result, save_predictions=True)
    assert (run_dir / "evaluation.json").is_file()
    assert len((run_dir / "predictions.jsonl").read_text().splitlines()) == 1


def test_winoground_rejects_non_matching_caption_token_multisets(tmp_path: Path) -> None:
    image_root = _images(tmp_path)
    annotation = tmp_path / "winoground.jsonl"
    _write_jsonl(
        annotation,
        [
            {
                "sample_id": "0",
                "image_0": "a.jpg",
                "image_1": "b.jpg",
                "caption_0": "a red object",
                "caption_1": "b blue object",
                "category": "winoground",
            }
        ],
    )
    config = _config(tmp_path, "winoground", annotation, image_root)
    config["evaluation"]["require_caption_token_multiset_match"] = True

    try:
        evaluate_baseline(config, encoder=DiagnosticEncoder(), project_root=tmp_path)
    except ValueError as exc:
        assert "identical caption token multisets" in str(exc)
    else:
        raise AssertionError("expected Winoground token-multiset validation to fail")


def _images(tmp_path: Path) -> Path:
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (2, 2)).save(image_root / "a.jpg")
    Image.new("RGB", (2, 2)).save(image_root / "b.jpg")
    return image_root


def _config(
    tmp_path: Path,
    dataset: str,
    annotation: Path,
    image_root: Path,
) -> dict[str, Any]:
    manifest = tmp_path / "dataset.yaml"
    manifest.write_text("manifest\n", encoding="utf-8")
    return {
        "experiment": {"name": "diagnostic", "seed": 42, "output_dir": str(tmp_path / "out")},
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
            "protocol": "synthetic-diagnostic-v1",
            "image_batch_size": 2,
            "text_batch_size": 2,
            "ranking_batch_size": 2,
            "recall_at": [1],
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
