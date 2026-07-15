from recoalign.analysis.results import collect_runs, render_markdown_table
from recoalign.experiments.records import create_run, finalize_run

CONFIG = {
    "experiment": {"name": "table", "seed": 1, "output_dir": "unused"},
    "model": {"framework": "open_clip", "name": "ViT-B-32", "pretrained": "test"},
    "data": {"dataset": "toy", "root": "data/toy", "split": "test"},
    "evaluation": {"recall_at": [1]},
    "training": {"enabled": False},
}


def test_only_reportable_runs_enter_tables(tmp_path) -> None:
    reportable = create_run(CONFIG, output_root=tmp_path, run_id="reportable")
    finalize_run(reportable, {"R@1": 60}, status="reportable")
    pilot = create_run(CONFIG, output_root=tmp_path, run_id="pilot")
    finalize_run(pilot, {"R@1": 99}, status="pilot")

    records = collect_runs(tmp_path)
    table = render_markdown_table(records, ["R@1"])
    assert "reportable" in table
    assert "pilot" not in table
    assert "60.0000" in table
