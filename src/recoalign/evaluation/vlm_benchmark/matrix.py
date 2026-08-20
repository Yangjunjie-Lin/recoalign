"""Complete evaluation matrix planning and non-selective coverage accounting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .adapters import inspect_benchmark_availability
from .methods import validate_method

REQUIRED_ABLATIONS = {
    "architecture": {
        "recoalign",
        "no_structure_token",
        "random_structure_token",
        "fixed_graph_encoder",
    },
    "training": {
        "recoalign",
        "no_structural_loss",
        "no_semantic_loss",
        "no_reasoning_alignment",
    },
    "data": {"synthetic_only", "real_only", "mixed_training"},
}


@dataclass(frozen=True)
class MatrixCell:
    model: str
    benchmark: str
    method: str
    seed: int
    status: str
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "benchmark": self.benchmark,
            "method": self.method,
            "seed": self.seed,
            "status": self.status,
            "reason": self.reason,
        }


def load_matrix_config(
    path: str | Path = "configs/benchmarks/comprehensive_matrix.yaml",
) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("comprehensive matrix config root must be a mapping")
    validate_evaluation_matrix(payload)
    return payload


def validate_evaluation_matrix(config: dict[str, Any]) -> dict[str, Any]:
    for key in ("models", "benchmarks", "methods", "seeds", "protocol_lock"):
        if key not in config:
            raise ValueError(f"comprehensive matrix is missing {key}")
    if not isinstance(config["models"], list) or not config["models"]:
        raise ValueError("matrix models must be a non-empty list")
    if not isinstance(config["benchmarks"], list) or not config["benchmarks"]:
        raise ValueError("matrix benchmarks must be a non-empty list")
    if not isinstance(config["methods"], list) or not config["methods"]:
        raise ValueError("matrix methods must be a non-empty list")
    seeds = config["seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) < 3
        or len(set(seeds)) != len(seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ValueError("matrix requires at least three unique non-negative seeds")
    protocol = config["protocol_lock"]
    if not isinstance(protocol, dict) or protocol.get("allow_split_override") is not False:
        raise ValueError("matrix protocol lock must prohibit split overrides")
    for model in config["models"]:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("matrix model names must be non-empty strings")
    for method in config["methods"]:
        if isinstance(method, str):
            validate_method(method)
        elif isinstance(method, dict):
            validate_method(str(method.get("name", "")))
        else:
            raise ValueError("matrix methods must be names or mappings")
    for benchmark in config["benchmarks"]:
        if not isinstance(benchmark, dict):
            raise ValueError("matrix benchmark rows must be mappings")
        required = (
            "name",
            "category",
            "format",
            "manifest",
            "annotation_file",
            "image_root",
            "split",
        )
        missing = [field for field in required if not str(benchmark.get(field, "")).strip()]
        if missing:
            raise ValueError(
                f"matrix benchmark {benchmark.get('name', '<unknown>')} is missing "
                f"{', '.join(missing)}"
            )
    return {
        "valid": True,
        "models": list(config["models"]),
        "benchmarks": [str(row["name"]) for row in config["benchmarks"]],
        "methods": [
            str(row if isinstance(row, str) else row["name"]) for row in config["methods"]
        ],
        "seeds": list(seeds),
        "minimum_cells": len(config["models"])
        * len(config["benchmarks"])
        * len(config["methods"])
        * len(seeds),
        "no_selective_reporting": True,
    }


def build_evaluation_matrix(
    config: dict[str, Any], *, project_root: str | Path | None = None
) -> dict[str, Any]:
    validate_evaluation_matrix(config)
    methods = [
        row if isinstance(row, dict) else {"name": row}
        for row in config["methods"]
    ]
    cells: list[MatrixCell] = []
    benchmark_lookup = {str(row["name"]): row for row in config["benchmarks"]}
    for model in config["models"]:
        for benchmark_name, benchmark in benchmark_lookup.items():
            availability = inspect_benchmark_availability(
                {"benchmark": benchmark}, project_root=project_root
            )
            for method in methods:
                method_name = str(method["name"])
                method_blocked = method.get("status") == "pending"
                for seed in config["seeds"]:
                    if method_blocked:
                        status, reason = "blocked", method.get("reason", "method is pending")
                    elif not availability["available"]:
                        status, reason = "blocked", "benchmark manifest/data unavailable"
                    else:
                        status, reason = "planned", None
                    cells.append(
                        MatrixCell(model, benchmark_name, method_name, seed, status, reason)
                    )
    counts: dict[str, int] = {}
    for cell in cells:
        counts[cell.status] = counts.get(cell.status, 0) + 1
    return {
        "schema_version": 1,
        "models": list(config["models"]),
        "benchmarks": list(benchmark_lookup),
        "methods": [method["name"] for method in methods],
        "seeds": list(config["seeds"]),
        "minimum_cells": len(cells),
        "counts": counts,
        "cells": [cell.to_dict() for cell in cells],
        "benchmark_availability": {
            name: inspect_benchmark_availability(
                {"benchmark": benchmark}, project_root=project_root
            )
            for name, benchmark in benchmark_lookup.items()
        },
        "no_selective_reporting": True,
        "scientific_decision": "INCONCLUSIVE",
    }


def load_ablation_matrix(
    path: str | Path = "configs/benchmarks/ablation_matrix.yaml",
) -> dict[str, Any]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ablation matrix root must be a mapping")
    for category, required in REQUIRED_ABLATIONS.items():
        values = payload.get(category)
        if not isinstance(values, list) or set(values) != required:
            raise ValueError(
                f"ablation matrix {category} must contain exactly {', '.join(sorted(required))}"
            )
        for method in values:
            validate_method(str(method))
    controls = payload.get("control", {})
    if not isinstance(controls, dict) or not all(
        controls.get(field) is True
        for field in (
            "shared_models",
            "shared_benchmarks",
            "shared_seeds",
            "shared_decoding",
            "shared_prompts",
            "single_factor_changes_only",
        )
    ):
        raise ValueError("ablation matrix must lock every fairness control")
    if payload.get("report_failed_cells") is not True:
        raise ValueError("ablation matrix must report failed cells")
    base = load_matrix_config(str(payload.get("base_matrix", "")))
    method_names = sorted(set().union(*REQUIRED_ABLATIONS.values()))
    base["methods"] = [{"name": name} for name in method_names]
    validate_evaluation_matrix(base)
    return {"definition": payload, "resolved_matrix": base}


def materialize_matrix_configs(
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write every registered cell config so execution cannot select favorable cells."""

    validation = validate_evaluation_matrix(config)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"matrix config output is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    methods = [row if isinstance(row, dict) else {"name": row} for row in config["methods"]]
    count = 0
    for model in config["models"]:
        for benchmark in config["benchmarks"]:
            for method in methods:
                for seed in config["seeds"]:
                    payload = {
                        "schema_version": 1,
                        "output_root": config.get("reporting", {}).get(
                            "results_root", "outputs/comprehensive"
                        ),
                        "model": {"name": model},
                        "method": dict(method),
                        "benchmark": dict(benchmark),
                        "seed": seed,
                        "prompt": dict(config.get("prompt", {})),
                        "generation": dict(config.get("generation", {})),
                        "statistics": dict(config.get("statistics", {})),
                        "protocol_lock": dict(config.get("protocol_lock", {})),
                    }
                    path = (
                        destination
                        / str(model)
                        / str(benchmark["name"])
                        / str(method["name"])
                        / f"seed_{seed}.yaml"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
                    count += 1
    manifest = {
        "schema_version": 1,
        "config_count": count,
        "expected_count": validation["minimum_cells"],
        "complete": count == validation["minimum_cells"],
        "selective_filtering": False,
    }
    (destination / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    return manifest


__all__ = [
    "MatrixCell",
    "build_evaluation_matrix",
    "load_ablation_matrix",
    "load_matrix_config",
    "materialize_matrix_configs",
    "validate_evaluation_matrix",
]
