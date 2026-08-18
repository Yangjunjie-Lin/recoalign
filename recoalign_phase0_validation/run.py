from __future__ import annotations

import argparse
from pathlib import Path

from phase0 import ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ReCoAlign Phase 0 selective semantic preservation experiment."
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "llava", "clip"),
        default="auto",
        help="auto uses LLaVA's vision+projector on CUDA and CLIP fallback on CPU.",
    )
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or a torch device such as cuda:0"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a small engineering check in _smoke_workspace; never eligible for GO.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.work_dir.resolve()
    kwargs = {}
    if args.smoke_test:
        root = root / "_smoke_workspace"
        kwargs = {
            "n_train": 144,
            "n_test": 72,
            "n_control_pairs": 12,
            "bootstrap_samples": 100,
            "probe_max_iter": 80,
            "protocol_mode": "smoke",
        }
    config = ExperimentConfig(
        root=root,
        backend=args.backend,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        force_data=args.force_data,
        force_features=args.force_features,
        **kwargs,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
