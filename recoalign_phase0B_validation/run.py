from __future__ import annotations

import argparse
from pathlib import Path

from phase0b import ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen ReCoAlign Phase 0-B semantic-utilization experiment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "protocol.json",
    )
    parser.add_argument("--force-vision", action="store_true")
    parser.add_argument("--force-llm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    if args.force_vision or args.force_llm:
        values = config.to_dict()
        values.pop("scientific_protocol_valid")
        values["root"] = Path(values["root"])
        values["source_phase0_dir"] = Path(values["source_phase0_dir"])
        values["model_dir"] = Path(values["model_dir"])
        values["layer_indices"] = tuple(values["layer_indices"])
        values["force_vision"] = args.force_vision
        values["force_llm"] = args.force_llm
        config = ExperimentConfig(**values)
    run_experiment(config)


if __name__ == "__main__":
    main()
