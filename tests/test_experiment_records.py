import json

from recoalign.experiments.records import create_run, finalize_run

CONFIG = {
    "experiment": {"name": "unit test", "seed": 7, "output_dir": "unused"},
    "model": {
        "framework": "open_clip",
        "name": "ViT-B-32",
        "pretrained": "test-checkpoint",
        "precision": "fp32",
    },
    "data": {"dataset": "toy", "root": "data/toy", "split": "test"},
    "evaluation": {"recall_at": [1, 5]},
    "training": {"enabled": False},
}


def test_run_lifecycle(tmp_path) -> None:
    run_dir = create_run(CONFIG, output_root=tmp_path, run_id="fixed-run")
    assert (run_dir / "config.resolved.yaml").is_file()
    assert (run_dir / "environment.json").is_file()

    record = finalize_run(run_dir, {"R@1": 50, "mean_recall": 75.25}, status="complete")
    assert record["status"] == "complete"

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics == {"R@1": 50.0, "mean_recall": 75.25}
