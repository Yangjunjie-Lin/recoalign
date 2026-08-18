from __future__ import annotations

import json
import os
import random
from typing import Any

import numpy as np
import torch

from .config import ExperimentConfig
from .data import generate_dataset
from .features import extract_all_features
from .probes import run_probes
from .reporting import write_results


def _set_reproducibility(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    config.validate()
    _set_reproducibility(config.seed)
    config.root.mkdir(parents=True, exist_ok=True)
    generate_dataset(config)
    feature_info = extract_all_features(config)
    probe_results = run_probes(config)
    decision = write_results(config, feature_info, probe_results)
    summary = {
        "decision": decision,
        "results_dir": str(config.results_dir.resolve()),
        "feature_info": feature_info,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary
