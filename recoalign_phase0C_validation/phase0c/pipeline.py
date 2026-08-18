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

from .analysis import (
    make_decision,
    run_availability_probes,
    summarize_attention,
    summarize_patching,
)
from .data import load_registered_records, read_jsonl, sha256_file
from .model import extract_behavior_and_hidden, run_activation_patching
from .reporting import write_outputs


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def run_experiment(
    config: Any,
    *,
    force_behavior: bool = False,
    force_patch: bool = False,
    recompute_test: bool = False,
) -> dict[str, Any]:
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
    protocol_path = config.configs_dir / "protocol.json"
    protocol_sha256 = sha256_file(protocol_path)
    records, source_info = load_registered_records(config)
    behavior_manifest = extract_behavior_and_hidden(
        config,
        records,
        source_info,
        force=force_behavior,
        recompute_test=recompute_test,
    )
    behavior_rows = read_jsonl(config.features_dir / "behavior_cache.jsonl")
    attention_rows = read_jsonl(config.features_dir / "attention_cache.jsonl")
    availability_rows, availability_predictions, availability_statistics = (
        run_availability_probes(config, records, behavior_rows)
    )
    patch_rows, patch_manifest = run_activation_patching(
        config,
        records,
        source_info,
        behavior_rows,
        availability_predictions,
        force=force_patch,
    )
    attention_statistics = summarize_attention(config, attention_rows)
    patch_statistics = summarize_patching(config, patch_rows)
    integrity = {
        "behavior_complete": bool(
            behavior_manifest.get("complete") and len(behavior_rows) == len(records)
        ),
        "attention_complete": bool(
            behavior_manifest.get("attention_rows") == config.n_test * 32
            and len(attention_rows) == config.n_test * 32
        ),
        "patch_complete": bool(patch_manifest.get("complete")),
        "patch_baseline_reproduced": bool(
            patch_manifest.get("baseline_mismatches", 0) == 0
        ),
        "source_index_sha256": source_info["index_sha256"],
        "protocol_sha256": protocol_sha256,
    }
    decision = make_decision(config, availability_rows, patch_statistics, integrity)
    decision.update(
        {
            "protocol_sha256": protocol_sha256,
            "source_index_sha256": source_info["index_sha256"],
            "model_checkpoint": "liuhaotian/llava-v1.5-7b@4481d270cc22fd5c4d1bb5df129622006ccd9234",
            "runtime_precision": config.quantization,
        }
    )
    statistics = {
        "protocol": config.to_dict(),
        "protocol_sha256": protocol_sha256,
        "source": source_info,
        "integrity": integrity,
        "availability": availability_statistics,
        "attention": attention_statistics,
        "activation_patch": patch_statistics,
        "manifests": {
            "behavior": behavior_manifest,
            "activation_patch": patch_manifest,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    write_outputs(
        config,
        records,
        behavior_rows,
        availability_rows,
        availability_predictions,
        attention_rows,
        patch_rows,
        statistics,
        decision,
    )
    return {
        "decision": decision["decision"],
        "semantic_access_failure": decision["semantic_access_failure"],
        "route": decision["route"],
        "decision_path": str((config.results_dir / "decision.json").resolve()),
    }
