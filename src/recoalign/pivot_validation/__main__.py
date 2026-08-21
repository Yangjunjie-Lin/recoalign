"""Module entry point for preregistered PH001 validation."""

from __future__ import annotations

import argparse
import json

from .runner import DEFAULT_CONFIG, preflight_pivot_validation, run_pivot_validation


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m recoalign.pivot_validation")
    parser.add_argument("action", choices=("preflight", "pilot", "final"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    if args.action == "preflight":
        result = preflight_pivot_validation(args.config)
        summary = {"status": result["status"], "weights_loaded": result["weights_loaded"]}
    else:
        result = run_pivot_validation(args.config, stage=args.action)
        summary = {
            "stage": args.action,
            "decision": result["metrics"]["mechanism"]["decision"],
            "classification": result["metrics"]["mechanism"]["classification"],
            "prediction_count": result["metrics"]["prediction_count"],
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
