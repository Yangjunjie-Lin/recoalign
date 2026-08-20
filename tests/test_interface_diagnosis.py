from __future__ import annotations

import json
from pathlib import Path

from diagnosis.interface_diagnosis.probes import graph_score, linear_probe
from diagnosis.interface_diagnosis.runner import run_interface_diagnosis
from recoalign.cli import build_parser


def test_linear_probe_is_deterministic_and_reports_metrics() -> None:
    import numpy as np

    features = np.asarray(
        [[1.0, 0.0], [1.1, 0.0], [-1.0, 0.0], [-1.1, 0.0], [1.2, 0.0], [-1.2, 0.0]]
    )
    first = linear_probe(features, ["a", "a", "b", "b", "a", "b"], seed=11)
    second = linear_probe(features, ["a", "a", "b", "b", "a", "b"], seed=11)
    assert first == second
    assert first["status"] == "available"
    assert first["accuracy"] == 1.0


def test_graph_score_reports_node_relation_and_edit_metrics() -> None:
    graph = {
        "objects": [{"id": "a", "shape": "cube", "color": "red"}],
        "relations": [{"subject": "a", "relation": "left", "object": "b"}],
    }
    scored = graph_score(graph, graph)
    assert scored["node_f1"] == 1.0
    assert scored["relation_f1"] == 1.0
    assert scored["graph_edit_distance"] == 0.0


def test_exp004_reference_run_writes_diagnosis_bundle(tmp_path: Path) -> None:
    output = run_interface_diagnosis(
        model_name="reference", seeds=[101], output_dir=tmp_path / "exp004"
    )
    assert {"metrics.json", "analysis.json", "decision_report.yaml", "run.json"} <= {
        path.name for path in output.iterdir()
    }
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["experiment_id"] == "EXP004"
    assert metrics["decision"] == "INCONCLUSIVE"
    assert metrics["diagnosis"]["claim_status"] == "infrastructure_validation"
    assert "mean" in metrics["metrics"]["aggregate"]["diagnosis.sas"]
    assert (output / "seeds" / "101" / "predictions.jsonl").is_file()


def test_exp004_cli_entrypoint_is_registered() -> None:
    parser = build_parser()
    args = parser.parse_args(["run-interface-diagnosis", "--model", "reference", "--dry-run"])
    assert args.command == "run-interface-diagnosis"
    assert args.model == "reference"


def test_base_vlm_declares_diagnostic_representation_boundaries() -> None:
    from recoalign.models.vlm.base import BaseVLM

    for method in ("encode_visual_tokens", "encode_visual_hidden", "reconstruct_graph"):
        assert hasattr(BaseVLM, method)
