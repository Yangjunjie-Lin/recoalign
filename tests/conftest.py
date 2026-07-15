from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def research_config(tmp_path: Path) -> dict[str, Any]:
    data_root = tmp_path / "data"
    data_root.mkdir()
    content = b"recoalign-test\n"
    data_file = data_root / "sample.txt"
    data_file.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()

    dataset_manifest = tmp_path / "dataset.yaml"
    dataset_manifest.write_text(
        f"""schema_version: 1
name: toy
version: v1
source: local-test
license: test-only
splits:
  test: 1
files:
  - path: sample.txt
    bytes: {len(content)}
    sha256: {digest}
processing: null
notes: unit test
""",
        encoding="utf-8",
    )

    checkpoint_manifest = tmp_path / "checkpoint.yaml"
    checkpoint_manifest.write_text(
        """schema_version: 1
framework: synthetic
model: identity
identifier: fixture
source: local-test
license: test-only
files: []
notes: unit test
""",
        encoding="utf-8",
    )

    return {
        "experiment": {"name": "unit test", "seed": 7, "output_dir": "unused"},
        "model": {
            "framework": "synthetic",
            "name": "identity",
            "pretrained": "fixture",
            "manifest": str(checkpoint_manifest),
            "precision": "fp32",
        },
        "data": {
            "dataset": "toy",
            "root": str(data_root),
            "manifest": str(dataset_manifest),
            "split": "test",
        },
        "evaluation": {"recall_at": [1, 5]},
        "training": {"enabled": False},
    }
