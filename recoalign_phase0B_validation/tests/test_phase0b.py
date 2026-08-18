from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase0b.config import ExperimentConfig  # noqa: E402
from phase0b.data import _balanced_sample  # noqa: E402
from phase0b.decision import determine_decision  # noqa: E402
from phase0b.probes import SEMANTICS  # noqa: E402


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        root=tmp_path / "phase0b",
        source_phase0_dir=tmp_path / "phase0",
        model_dir=tmp_path / "model",
    )


def _probe_result(*, passing: bool) -> dict:
    rows = []
    for semantic in SEMANTICS:
        rows.append({"stage": "Za", "semantic": semantic, "accuracy": 0.95})
    drops = {}
    selectivity = {}
    for layer in (8, 16, 24, 32):
        stage = f"L{layer:02d}_visual"
        relation = 0.25 if passing and layer == 16 else 0.02
        composition = 0.30 if passing and layer == 16 else 0.03
        object_drop = 0.01
        drops[stage] = {
            "object": {"drop": object_drop, "ci95_low": 0.0, "ci95_high": 0.02},
            "attribute": {"drop": 0.01, "ci95_low": 0.0, "ci95_high": 0.02},
            "relation": {
                "drop": relation,
                "ci95_low": relation - 0.04,
                "ci95_high": relation + 0.04,
            },
            "composition": {
                "drop": composition,
                "ci95_low": composition - 0.04,
                "ci95_high": composition + 0.04,
            },
        }
        selectivity[stage] = {
            "relation": {"ci95_low": relation - object_drop - 0.04},
            "composition": {"ci95_low": composition - object_drop - 0.04},
        }
    return {
        "rows": rows,
        "drops": drops,
        "selectivity": selectivity,
        "identity": {"max_abs_difference": 0.001},
        "all_probes_converged": True,
    }


def test_registered_config_is_full_and_pre_registered(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.validate()
    assert config.scientific_protocol_valid()
    assert config.layer_indices == (0, 8, 16, 24, 32)
    assert config.min_specific_drop == pytest.approx(0.15)


def test_go_requires_same_layer_specific_degradation(tmp_path: Path) -> None:
    decision = determine_decision(_config(tmp_path), _probe_result(passing=True))
    assert decision["decision"] == "GO"
    assert decision["passing_layers"] == [16]
    assert decision["next_phase"] == "Causal Intervention design"


def test_null_result_is_no_go(tmp_path: Path) -> None:
    decision = determine_decision(_config(tmp_path), _probe_result(passing=False))
    assert decision["decision"] == "NO-GO"
    assert not decision["semantic_utilization_failure"]


def test_balanced_selection_has_equal_class_counts() -> None:
    records = []
    for class_index in range(72):
        for item in range(12):
            records.append(
                {
                    "split": "train",
                    "composition": f"class-{class_index:02d}",
                    "image_id": f"{class_index:02d}-{item:02d}",
                }
            )
    selected = _balanced_sample(records, split="train", count=720, seed=7)
    counts = {}
    for row in selected:
        counts[row["composition"]] = counts.get(row["composition"], 0) + 1
    assert len(selected) == 720
    assert set(counts.values()) == {10}


def test_protocol_json_is_valid() -> None:
    payload = json.loads((ROOT / "configs" / "protocol.json").read_text(encoding="utf-8"))
    assert payload["n_train"] == 720
    assert payload["n_test"] == 288
    assert payload["quantization"] == "nf4"
