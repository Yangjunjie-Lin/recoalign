#!/usr/bin/env python3
"""Recompute Winoground decisions and write local prediction-audit summaries."""

from __future__ import annotations

import argparse
import json

from recoalign.experiments.winoground_audit import audit_winoground_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit one completed Winoground run")
    parser.add_argument("run_dir")
    parser.add_argument("--expected-samples", type=int, default=400)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = audit_winoground_run(args.run_dir, expected_samples=args.expected_samples)
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
