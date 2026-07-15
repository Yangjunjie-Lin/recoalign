"""Command-line interface for Phase-0 research infrastructure."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from recoalign.analysis.results import collect_runs, render_markdown_table
from recoalign.config import ConfigError, config_digest, load_config
from recoalign.data.manifest import load_dataset_manifest, verify_dataset
from recoalign.experiments.records import RUN_STATUSES, create_run, finalize_run
from recoalign.reproducibility import atomic_write_json, collect_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recoalign")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="validate an experiment YAML file")
    validate.add_argument("config")

    environment = subparsers.add_parser("capture-environment", help="write environment metadata")
    environment.add_argument("--output", default="environment.json")

    initialize = subparsers.add_parser("init-run", help="create a self-contained run directory")
    initialize.add_argument("--config", required=True)
    initialize.add_argument("--output-root")
    initialize.add_argument("--run-id")

    finalize = subparsers.add_parser("finalize-run", help="attach metrics and finalize a run")
    finalize.add_argument("run_dir")
    finalize.add_argument("--metrics", required=True)
    finalize.add_argument("--status", choices=sorted(RUN_STATUSES), default="complete")
    finalize.add_argument("--notes")

    verify = subparsers.add_parser("verify-dataset", help="verify files declared by a manifest")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--root", required=True)

    table = subparsers.add_parser("build-table", help="build a Markdown table from completed runs")
    table.add_argument("--results-root", default="results")
    table.add_argument("--metrics", nargs="+", required=True)
    table.add_argument("--status", nargs="+", default=["reportable"])
    table.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate-config":
            config = load_config(args.config)
            print(f"valid: {args.config}")
            print(f"sha256: {config_digest(config)}")
            return 0

        if args.command == "capture-environment":
            atomic_write_json(args.output, collect_environment(Path.cwd()))
            print(args.output)
            return 0

        if args.command == "init-run":
            config = load_config(args.config)
            run_dir = create_run(
                config,
                config_path=args.config,
                output_root=args.output_root,
                run_id=args.run_id,
            )
            print(run_dir)
            return 0

        if args.command == "finalize-run":
            with Path(args.metrics).open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            finalize_run(args.run_dir, metrics, status=args.status, notes=args.notes)
            print(Path(args.run_dir) / "run.json")
            return 0

        if args.command == "verify-dataset":
            manifest = load_dataset_manifest(args.manifest)
            failures = verify_dataset(args.root, manifest)
            if failures:
                print("\n".join(failures))
                return 1
            print("dataset verification passed")
            return 0

        if args.command == "build-table":
            records = collect_runs(args.results_root, args.status)
            rendered = render_markdown_table(records, args.metrics)
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(rendered, encoding="utf-8")
            else:
                print(rendered, end="")
            return 0
    except (ConfigError, FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    return 2
