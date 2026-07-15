import json

from recoalign.analysis.results import collect_runs, render_markdown_table
from recoalign.experiments.records import create_run, finalize_run, promote_run
from recoalign.reproducibility import atomic_write_json


def _make_reportable(run_dir) -> None:
    environment_path = run_dir / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["git"].update(
        {
            "commit": "a" * 40,
            "branch": "test",
            "dirty": False,
            "untracked_count": 0,
            "diff_sha256": None,
        }
    )
    atomic_write_json(environment_path, environment)

    run_path = run_dir / "run.json"
    record = json.loads(run_path.read_text(encoding="utf-8"))
    record["git_commit"] = "a" * 40
    atomic_write_json(run_path, record)
    promote_run(run_dir, reviewed_by="Reviewer")


def test_only_reportable_runs_enter_tables(tmp_path, research_config) -> None:
    reportable = create_run(research_config, output_root=tmp_path, run_id="reportable")
    finalize_run(reportable, {"R@1": 60}, status="complete")
    _make_reportable(reportable)

    pilot = create_run(research_config, output_root=tmp_path, run_id="pilot")
    finalize_run(pilot, {"R@1": 99}, status="pilot")

    records = collect_runs(tmp_path)
    table = render_markdown_table(records, ["R@1"])
    assert "reportable" in table
    assert "pilot" not in table
    assert "60.0000" in table
