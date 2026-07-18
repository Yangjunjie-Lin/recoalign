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


@pytest.mark.parametrize(("completed_rows", "expected_exit"), [(0, 1), (399, 1), (400, 0)])
def test_require_complete_exit_status(
    tmp_path: Path,
    review_helper: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    completed_rows: int,
    expected_exit: int,
) -> None:
    run_dir, review_path = _review_fixture(tmp_path)
    _complete_review_rows(review_path, completed_rows)
    monkeypatch.setattr(review_helper, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review_winoground.py",
            "--run-dir",
            str(run_dir),
            "--review-csv",
            str(review_path),
            "--check-only",
            "--require-complete",
        ],
    )

    assert review_helper.main() == expected_exit
    summary = json.loads(capsys.readouterr().out)
    assert summary["completed_rows"] == completed_rows
    assert summary["remaining_rows"] == 400 - completed_rows
    assert summary["complete"] is (completed_rows == 400)


def test_check_only_remains_successful_for_incomplete_queue(
    tmp_path: Path,
    review_helper: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir, review_path = _review_fixture(tmp_path)
    monkeypatch.setattr(review_helper, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review_winoground.py",
            "--run-dir",
            str(run_dir),
            "--review-csv",
            str(review_path),
            "--check-only",
        ],
    )

    assert review_helper.main() == 0
    assert json.loads(capsys.readouterr().out)["complete"] is False


def test_invalid_review_row_exits_two(
    tmp_path: Path,
    review_helper: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir, review_path = _review_fixture(tmp_path)
    with review_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["mapping_checked"] = "false"
    _write_review_rows(review_path, rows)
    monkeypatch.setattr(review_helper, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review_winoground.py",
            "--run-dir",
            str(run_dir),
            "--review-csv",
            str(review_path),
            "--check-only",
            "--require-complete",
        ],
    )

    assert review_helper.main() == 2
    assert "mapping_checked must be true" in capsys.readouterr().out


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


def _complete_review_rows(path: Path, count: int) -> None:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows[:count]:
        row["mapping_checked"] = "true"
        row["visual_review_status"] = "pass"
        row["annotation_issue"] = "none"
    _write_review_rows(path, rows)


def _write_review_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
