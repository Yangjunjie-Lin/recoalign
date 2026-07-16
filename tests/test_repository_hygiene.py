from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "dataset",
    ["flickr30k", "mscoco", "sugarcrepe", "aro", "winoground", "bivlc"],
)
def test_real_dataset_roots_are_gitignored(dataset: str) -> None:
    probe = f"data/{dataset}/untracked-probe"
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", probe],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    assert result.returncode == 0, f"real dataset root is not ignored: data/{dataset}/"
