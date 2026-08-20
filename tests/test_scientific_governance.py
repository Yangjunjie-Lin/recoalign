from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from evaluation.statistics import (
    bootstrap_confidence_interval,
    multiple_seed_summary,
    paired_t_test,
    permutation_test,
)
from recoalign.governance import run_registered_experiment
from recoalign.research_registry import (
    get_registered_experiment,
    validate_research_registries,
)
from recoalign.schema_validation import validate_payload
from research.decision_engine import evaluate_decision


def test_registries_are_linked_and_all_initial_experiments_are_registered() -> None:
    report = validate_research_registries()
    assert report["valid"] is True
    assert report["hypothesis_ids"] == ["H001", "H002", "H003", "H004"]
    assert report["experiment_ids"] == ["EXP001", "EXP002", "EXP003", "EXP004"]


@pytest.mark.parametrize("experiment_id", ["EXP001", "EXP002", "EXP003"])
def test_registered_experiments_can_start_as_dry_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, experiment_id: str
) -> None:
    monkeypatch.setattr(
        "recoalign.governance.collect_environment",
        lambda _root: {"captured_at": "test", "git": {}},
    )
    run_dir = run_registered_experiment(
        experiment_id,
        output_root=tmp_path,
        run_id=f"dry-{experiment_id.lower()}",
        dry_run=True,
    )
    required = {
        "config.resolved.yaml",
        "command.txt",
        "environment.txt",
        "git_commit.txt",
        "seed.txt",
        "metrics.json",
        "log.txt",
        "manifest.json",
        "decision_report.yaml",
        "decision_report.md",
    }
    assert required <= {path.name for path in run_dir.iterdir()}
    result = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    validate_payload("experiment_result", result)
    assert result["experiment_id"] == experiment_id
    assert result["decision"] == "INCONCLUSIVE"
    assert result["status"] == "dry-run"
    report = yaml.safe_load((run_dir / "decision_report.yaml").read_text(encoding="utf-8"))
    validate_payload("decision_report", report)
    assert report["final_decision"] == "INCONCLUSIVE"


def test_statistical_primitives_are_deterministic_and_directional() -> None:
    values = [0.1, 0.2, 0.3, 0.4]
    assert bootstrap_confidence_interval(values, samples=200, seed=9) == (
        bootstrap_confidence_interval(values, samples=200, seed=9)
    )
    t_result = paired_t_test(values, [0.0] * len(values), alternative="greater")
    assert t_result["p_value"] < 0.05
    permutation = permutation_test(values, [0.0] * len(values), alternative="greater")
    assert permutation["exact"] is True
    assert 0.0 <= permutation["p_value"] <= 1.0


def test_decision_engine_applies_all_registered_gates() -> None:
    experiment = deepcopy(get_registered_experiment("EXP001"))
    experiment["model"]["scientific_decision_allowed"] = True
    summary = multiple_seed_summary([0.2] * 5, bootstrap_samples=100, seed=1)
    aggregate = {
        criterion["metric_path"]: summary
        for criterion in experiment["decision_rule"]["criteria"]
    }
    assertions = experiment["decision_rule"].get("required_manifest_assertions", {})
    result = {
        "run_id": "decision-test",
        "experiment_id": "EXP001",
        "hypothesis": "H001",
        "model": "frozen-reference",
        "dataset": "recoalign-synthetic-compositional-world@generator-v2",
        "status": "complete",
        "metrics": {"aggregate": aggregate},
        "integrity": {
            "registry_validated": True,
            "per_seed": [
                {"manifest_assertions": dict(assertions)} for _ in range(5)
            ],
        },
    }
    report = evaluate_decision(experiment, result)
    validate_payload("decision_report", report)
    assert report["final_decision"] == "GO"
    result["dataset"] = "unregistered@tampered"
    assert evaluate_decision(experiment, result)["final_decision"] == "INCONCLUSIVE"
    result["dataset"] = "recoalign-synthetic-compositional-world@generator-v2"
    first_path = experiment["decision_rule"]["criteria"][0]["metric_path"]
    result["metrics"]["aggregate"][first_path]["mean"] = 0.0
    assert evaluate_decision(experiment, result)["final_decision"] == "NO-GO"


def test_registry_distinguishes_scientific_and_runtime_conditions() -> None:
    assert get_registered_experiment("EXP001")["conditions"] == ["graph", "text"]
    assert get_registered_experiment("EXP002")["conditions"] == [
        "image only",
        "full graph",
        "partial graph",
        "corrupted graph",
    ]
    assert get_registered_experiment("EXP003")["conditions"] == [
        "seen composition",
        "unseen composition",
    ]


@pytest.mark.parametrize("experiment_id", ["EXP001", "EXP002", "EXP003"])
def test_registered_experiments_execute_and_record_yaml_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, experiment_id: str
) -> None:
    registration = get_registered_experiment(experiment_id)

    def fake_run_config(_config: Path, *, output_dir: Path, seed: int) -> dict:
        del output_dir, seed
        metrics: dict = {}
        for criterion in registration["decision_rule"]["criteria"]:
            cursor = metrics
            parts = criterion["metric_path"].split(".")
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = 0.2
        return metrics

    def fake_integrity(_seed_dir: Path, seed: int, registered: dict) -> dict:
        return {
            "seed": seed,
            "manifest_assertions": dict(
                registered["decision_rule"].get("required_manifest_assertions", {})
            ),
        }

    monkeypatch.setattr("recoalign.governance.run_config", fake_run_config)
    monkeypatch.setattr("recoalign.governance._seed_integrity", fake_integrity)
    monkeypatch.setattr(
        "recoalign.governance.collect_environment",
        lambda _root: {"captured_at": "test", "git": {}},
    )
    run_dir = run_registered_experiment(
        experiment_id,
        output_root=tmp_path,
        run_id=f"execute-{experiment_id.lower()}",
    )
    report = yaml.safe_load((run_dir / "decision_report.yaml").read_text(encoding="utf-8"))
    validate_payload("decision_report", report)
    assert report["experiment_id"] == experiment_id
    assert report["final_decision"] == "INCONCLUSIVE"
    assert {
        "experiment_id",
        "evidence",
        "statistical_result",
        "robustness_check",
        "final_decision",
        "next_action",
    } <= report.keys()
    assert (run_dir / "decision_report.md").is_file()


def test_result_schema_example_is_valid() -> None:
    path = Path("results/schema/experiment_result.json")
    validate_payload("experiment_result", json.loads(path.read_text(encoding="utf-8")))


def test_unregistered_config_cannot_bypass_preregistration(tmp_path: Path) -> None:
    config = tmp_path / "copied.yaml"
    config.write_text(Path("configs/graph_vs_text.yaml").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="not preregistered"):
        run_registered_experiment("EXP001", config_path=config, output_root=tmp_path)
