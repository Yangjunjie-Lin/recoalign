"""Execute any registered Phase-1 protocol from its resolved YAML configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from evaluation.metrics import write_metrics
from experiments.runtime import (
    build_generator,
    build_vlm,
    load_config,
    run_conditions,
    write_run,
)


def run_config(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run one configured protocol; all scientific variables come from YAML."""

    config = deepcopy(load_config(config_path))
    if output_dir is not None:
        config["experiment"]["output_dir"] = str(Path(output_dir))
    if seed is not None:
        config["experiment"]["seed"] = seed
        config["model"]["seed"] = seed

    protocol = config["experiment"].get("protocol", config["experiment"]["name"])
    if protocol == "graph_vs_text":
        from experiments.graph_vs_text.runner import run_seed_config

        return run_seed_config(
            config,
            output_dir=Path(config["experiment"]["output_dir"]),
            seed=int(config["experiment"]["seed"]),
        )
    if protocol == "graph_ablation":
        from experiments.graph_ablation.runner import run_seed_config

        return run_seed_config(
            config,
            output_dir=Path(config["experiment"]["output_dir"]),
            seed=int(config["experiment"]["seed"]),
        )
    if protocol == "ood_composition":
        from experiments.ood_composition.runner import run_seed_config

        return run_seed_config(
            config,
            output_dir=Path(config["experiment"]["output_dir"]),
            seed=int(config["experiment"]["seed"]),
        )
    raise ValueError(f"unsupported structured experiment protocol: {protocol}")


def _run_ood_composition(
    config: dict[str, Any], conditions: tuple[str, ...]
) -> dict[str, Any]:
    generator = build_generator(config)
    output = Path(config["experiment"]["output_dir"])
    synthetic = config["synthetic"]
    seed = int(config["experiment"]["seed"])
    train, test = generator.generate_ood_splits(
        int(synthetic["train_count"]),
        int(synthetic["test_count"]),
        output_dir=output / "dataset",
        seed=seed,
    )
    rows = run_conditions(test, build_vlm(config), conditions)
    for row in rows:
        row["evaluation_split"] = config.get("data", {}).get("split", "ood_test")
    metrics = write_run(
        str(config["experiment"]["name"]), config, test, rows, output_dir=output
    )
    metrics["train_count"] = len(train)
    metrics["test_count"] = len(test)
    write_metrics(output / "metrics.json", metrics)
    return metrics
