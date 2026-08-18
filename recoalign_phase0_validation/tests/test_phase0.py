from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from phase0.config import ExperimentConfig
from phase0.data import generate_dataset, read_jsonl
from phase0.reporting import determine_decision
from phase0.statistics import bootstrap_drop_comparison


def test_dataset_has_balanced_classes_and_isolated_pairs(tmp_path: Path) -> None:
    config = ExperimentConfig(
        root=tmp_path,
        n_train=144,
        n_test=72,
        n_control_pairs=4,
        bootstrap_samples=20,
        protocol_mode="smoke",
    )
    generate_dataset(config)
    records = read_jsonl(tmp_path / "synthetic_dataset" / "metadata.jsonl")
    assert len(records) == 216
    assert len({row["composition"] for row in records if row["split"] == "train"}) == 72
    assert len({row["composition"] for row in records if row["split"] == "test"}) == 72
    required = {"object1", "object2", "attribute1", "attribute2", "relation", "composition"}
    assert required <= records[0].keys()

    controls = read_jsonl(tmp_path / "synthetic_dataset" / "control_metadata.jsonl")
    assert len(controls) == 8
    for index in range(0, len(controls), 2):
        first, second = controls[index : index + 2]
        assert first["pair_id"] == second["pair_id"]
        for key in ("object1", "object2", "attribute1", "attribute2"):
            assert first[key] == second[key]
        assert (first["relation"], second["relation"]) in {
            ("left", "right"),
            ("above", "below"),
        }
    manifest = json.loads((tmp_path / "synthetic_dataset" / "manifest.json").read_text())
    assert manifest["composition_classes"] == 72


def test_bootstrap_relation_vs_object_is_paired_and_deterministic() -> None:
    object_zv = np.ones(100, dtype=bool)
    object_za = np.ones(100, dtype=bool)
    relation_zv = np.ones(100, dtype=bool)
    relation_za = np.zeros(100, dtype=bool)
    first = bootstrap_drop_comparison(
        object_zv=object_zv,
        object_za=object_za,
        relation_zv=relation_zv,
        relation_za=relation_za,
        samples=100,
        seed=7,
    )
    second = bootstrap_drop_comparison(
        object_zv=object_zv,
        object_za=object_za,
        relation_zv=relation_zv,
        relation_za=relation_za,
        samples=100,
        seed=7,
    )
    assert first == second
    assert first["observed_difference"] == 1.0
    assert first["p_one_sided"] < 0.05


def test_go_requires_all_registered_gates(tmp_path: Path) -> None:
    config = ExperimentConfig(root=tmp_path)
    payload = {
        "drops": {
            "object": {"drop": 0.05},
            "attribute": {"drop": 0.10},
            "relation": {"drop": 0.30},
            "composition": {"drop": 0.31},
        },
        "object_vs_relation": {"p_one_sided": 0.01},
        "semantic_isolation_drop": {"drop": 0.27, "ci95_low": 0.20},
    }
    decision = determine_decision(config, payload)
    assert decision["decision"] == "GO"

    payload["semantic_isolation_drop"]["ci95_low"] = -0.01
    decision = determine_decision(config, payload)
    assert decision["decision"] == "NO-GO"
