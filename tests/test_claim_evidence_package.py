from __future__ import annotations

import gzip
import hashlib
import re
from pathlib import Path

import yaml


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_published_claim_evidence_matches_frozen_manifest() -> None:
    root = Path("reports/paper_evidence")
    manifest = yaml.safe_load((root / "artifact_manifest.yaml").read_text(encoding="utf-8"))

    assert manifest["scientific_decision"] == "NO-GO"
    assert set(manifest["experiments"]) == {"EXP001", "EXP002", "EXP003", "EXP004"}

    for experiment, record in manifest["experiments"].items():
        experiment_root = root / experiment
        for artifact_name in ("metrics", "decision_report"):
            artifact = record[artifact_name]
            path = experiment_root / Path(artifact["path"]).name
            payload = path.read_bytes()
            assert len(payload) == artifact["bytes"]
            assert _sha256(payload) == artifact["sha256"]

        predictions = record["predictions"]
        path = experiment_root / Path(predictions["path"]).name
        compressed = path.read_bytes()
        decompressed = gzip.decompress(compressed)
        assert len(compressed) == predictions["bytes"]
        assert _sha256(compressed) == predictions["sha256"]
        assert len(decompressed) == predictions["decompressed_bytes"]
        assert _sha256(decompressed) == predictions["decompressed_sha256"]
        assert len(decompressed.splitlines()) == record["prediction_lines"]


def test_failed_exp003_remains_unpromoted() -> None:
    root = Path("reports/paper_evidence/EXP003")
    result = yaml.safe_load((root / "metrics.json").read_text(encoding="utf-8"))
    decision = yaml.safe_load((root / "decision_report.yaml").read_text(encoding="utf-8"))

    assert result["status"] == "failed"
    assert result["decision"] == "INCONCLUSIVE"
    assert decision["final_decision"] == "INCONCLUSIVE"
    assert (root / "seed_20260818_metrics_unpromoted.json").is_file()
    assert (root / "seed_20260818_predictions_unpromoted.jsonl.gz").is_file()


def test_model_manifest_matches_frozen_evidence_identity() -> None:
    root = Path("reports/paper_evidence")
    model = yaml.safe_load((root / "model_manifest.yaml").read_text(encoding="utf-8"))
    evidence = yaml.safe_load((root / "artifact_manifest.yaml").read_text(encoding="utf-8"))

    assert model["identifier"] == evidence["model"]["identifier"]
    assert model["revision"] == evidence["model"]["revision"]
    assert model["checkpoint_fingerprint"] == evidence["model"]["checkpoint_fingerprint"]
    assert model["verification"]["passed"] is True
    assert model["verification"]["files_matching_bytes_and_sha256"] == 9


def test_published_claim_evidence_has_no_machine_local_paths() -> None:
    root = Path("reports/paper_evidence")
    local_path = re.compile(r"(?i)(?:\b[a-z]:[\\/]|/(?:home|Users)/)")

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix in {".gz", ".png"}:
            continue
        assert local_path.search(path.read_text(encoding="utf-8")) is None, path
