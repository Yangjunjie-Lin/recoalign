from __future__ import annotations

import json
import os
import platform
import random
from typing import Any

import numpy as np
import scipy
import sklearn
import torch

from .config import ExperimentConfig
from .data import select_protocol_records
from .features import extract_llm_features, extract_projected_tokens
from .probes import run_probes
from .reporting import write_results


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    # This must be set before the first CUDA BLAS operation. PyTorch otherwise
    # only warns under deterministic_algorithms(warn_only=True).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    config.validate()
    _seed_everything(config.seed)
    for directory in (
        config.root,
        config.configs_dir,
        config.features_dir,
        config.results_dir,
        config.figures_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    records = select_protocol_records(config)
    vision_info = extract_projected_tokens(config, records)
    llm_info = extract_llm_features(config, records)
    probe_results = run_probes(config, records)
    decision = write_results(
        config,
        {
            "vision_projector": vision_info,
            "llm": llm_info,
            "analysis_environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            },
        },
        probe_results,
    )
    summary = {
        "decision": decision["decision"],
        "semantic_utilization_failure": decision["semantic_utilization_failure"],
        "results_dir": str(config.results_dir.resolve()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary
