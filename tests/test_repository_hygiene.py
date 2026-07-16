from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _git_check_ignore(path: str) -> subprocess.CompletedProcess[str]:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is required to validate repository ignore policy")
    return subprocess.run(
        [git, "check-ignore", "--quiet", "--no-index", path],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


@pytest.mark.parametrize(
    "dataset",
    ["flickr30k", "mscoco", "sugarcrepe", "aro", "winoground", "bivlc"],
)
def test_real_dataset_roots_are_gitignored(dataset: str) -> None:
    probe = f"data/{dataset}/untracked-probe"
    result = _git_check_ignore(probe)
    assert result.returncode == 0, f"real dataset root is not ignored: data/{dataset}/"


@pytest.mark.parametrize(
    "probe",
    [
        "manifests/datasets/policy-probe.example.yaml",
        "manifests/datasets/README.md",
        "reports/policy-probe.md",
        "tests/policy_probe.py",
        "configs/policy-probe.yaml",
        "docs/policy-probe.md",
    ],
)
def test_research_policy_files_remain_trackable(probe: str) -> None:
    result = _git_check_ignore(probe)
    assert result.returncode == 1, f"research policy path is unexpectedly ignored: {probe}"
