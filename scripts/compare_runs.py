#!/usr/bin/env python3
"""Compare a canonical baseline run with an independent no-cache rerun."""

from __future__ import annotations

import argparse
import json

from recoalign.experiments.run_comparison import compare_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare cached and no-cache run outputs")
    parser.add_argument("cached_run")
    parser.add_argument("no_cache_run")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = compare_runs(
            args.cached_run,
            args.no_cache_run,
            atol=args.atol,
            rtol=args.rtol,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
