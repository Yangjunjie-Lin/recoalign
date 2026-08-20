#!/usr/bin/env python3
"""Recompute Winoground decisions and write local prediction-audit summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from recoalign.experiments.winoground_audit import audit_winoground_run
from recoalign.schema_validation import repository_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit one completed Winoground run")
    parser.add_argument("run_dir")
    parser.add_argument("--expected-samples", type=int, default=400)
    parser.add_argument(
        "--annotation-file",
        help=(
            "normalized annotation JSONL; defaults to data.annotation_file in the "
            "run's config.resolved.yaml"
        ),
    )
    return parser.parse_args()


def resolve_annotation_path(run_dir: str | Path, override: str | None = None) -> Path:
    if override:
        path = Path(override)
    else:
        config_path = Path(run_dir) / "config.resolved.yaml"
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, dict):
            raise ValueError("resolved config root must be a mapping")
        data = config.get("data")
        annotation = data.get("annotation_file") if isinstance(data, dict) else None
        if not isinstance(annotation, str) or not annotation.strip():
            raise ValueError("resolved config does not declare data.annotation_file")
        path = Path(annotation)
    if not path.is_absolute():
        path = repository_root() / path
    return path


def main() -> int:
    args = parse_args()
    try:
        annotation_path = resolve_annotation_path(args.run_dir, args.annotation_file)
        summary = audit_winoground_run(
            args.run_dir,
            expected_samples=args.expected_samples,
            annotation_path=annotation_path,
            require_annotation_alignment=True,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
