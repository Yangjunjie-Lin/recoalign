from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml


@pytest.fixture
def review_helper() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "review_winoground.py"
    spec = importlib.util.spec_from_file_location("review_winoground", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_review_workspace_checks_queue_and_saves_without_defaults(
    tmp_path: Path, review_helper: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, review_path = _review_fixture(tmp_path)
    monkeypatch.setattr(review_helper, "REPOSITORY_ROOT", tmp_path)
    workspace = review_helper.ReviewWorkspace.load(run_dir, review_path)

    assert workspace.summary()["completed_rows"] == 0
    assert workspace.summary()["remaining_rows"] == 400
    assert workspace.public_items()[0]["review_group"] == "both_directions_incorrect"

    workspace.save(
        sample_id="winoground-000000",
        mapping_checked=True,
        visual_review_status="pass",
        annotation_issue="none",
        notes="",
    )

    assert workspace.summary()["completed_rows"] == 1
    with review_path.open(encoding="utf-8-sig", newline="") as handle:
        first = next(csv.DictReader(handle))
    assert first["mapping_checked"] == "true"
    assert first["visual_review_status"] == "pass"
    assert first["annotation_issue"] == "none"

    with pytest.raises(review_helper.ReviewConflict, match="read-only"):
        workspace.save(
            sample_id="winoground-000000",
            mapping_checked=True,
            visual_review_status="pass",
            annotation_issue="none",
            notes="changed",
        )


def test_review_workspace_requires_notes_for_uncertain(
    tmp_path: Path, review_helper: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, review_path = _review_fixture(tmp_path)
    monkeypatch.setattr(review_helper, "REPOSITORY_ROOT", tmp_path)
    workspace = review_helper.ReviewWorkspace.load(run_dir, review_path)

    with pytest.raises(ValueError, match="uncertain rows require notes"):
        workspace.save(
            sample_id="winoground-000000",
            mapping_checked=True,
            visual_review_status="uncertain",
            annotation_issue="possible",
            notes="",
        )


def _review_fixture(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    image_root = tmp_path / "images"
    run_dir.mkdir()
    image_root.mkdir()
    annotations = tmp_path / "annotations.jsonl"
    predictions = run_dir / "predictions.jsonl"
    annotation_rows = []
    prediction_rows = []
    for index in range(400):
        sample_id = f"winoground-{index:06d}"
        image_0 = f"{index:06d}_image_0.png"
        image_1 = f"{index:06d}_image_1.png"
        (image_root / image_0).write_bytes(b"image-0")
        (image_root / image_1).write_bytes(b"image-1")
        annotation_rows.append(
            {
                "sample_id": sample_id,
                "image_0": image_0,
                "image_1": image_1,
                "caption_0": "caption zero",
                "caption_1": "caption one",
                "tags": ["synthetic"],
            }
        )
        prediction_rows.append(
            {
                "sample_id": sample_id,
                "scores": [0.1, 0.2, 0.3, 0.4],
                "image_to_text_correct": False,
                "text_to_image_correct": False,
                "group_correct": False,
                "tie": False,
            }
        )
    _write_jsonl(annotations, annotation_rows)
    _write_jsonl(predictions, prediction_rows)
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "data": {
                    "annotation_file": str(annotations),
                    "image_root": str(image_root),
                }
            }
        ),
        encoding="utf-8",
    )
    review_path = tmp_path / "review.csv"
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "review_group",
                "mapping_checked",
                "visual_review_status",
                "annotation_issue",
                "notes",
            ]
        )
        for index in range(400):
            writer.writerow(
                [f"winoground-{index:06d}", "both_directions_incorrect", "", "", "", ""]
            )
    return run_dir, review_path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
