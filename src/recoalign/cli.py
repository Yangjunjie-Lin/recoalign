"""Command-line interface for reproducible research infrastructure."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from recoalign.analysis.results import collect_runs, render_markdown_table
from recoalign.config import ConfigError, config_digest, load_config
from recoalign.data.manifest import ManifestError, load_dataset_manifest, verify_dataset
from recoalign.data.preparation import prepare_coco, prepare_flickr30k, prepare_sugarcrepe
from recoalign.evaluation.baseline import evaluate_baseline, write_baseline_outputs
from recoalign.experiments.records import (
    FINALIZABLE_STATUSES,
    create_run,
    fail_run,
    finalize_run,
    promote_run,
)
from recoalign.reproducibility import atomic_write_json, collect_environment
from recoalign.schema_validation import SchemaValidationError


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

    baseline = subparsers.add_parser(
        "run-baseline",
        help="create, evaluate, and finalize one zero-shot baseline run",
    )
    baseline.add_argument("--config", required=True)
    baseline.add_argument("--output-root")
    baseline.add_argument("--run-id")
    baseline.add_argument("--no-cache", action="store_true")

    flickr = subparsers.add_parser(
        "prepare-flickr30k", help="normalize the Karpathy Flickr30K split"
    )
    flickr.add_argument("--karpathy-json", required=True)
    flickr.add_argument("--dataset-root", required=True)
    flickr.add_argument("--manifest-output", required=True)
    flickr.add_argument("--source", required=True)
    flickr.add_argument("--license", required=True, dest="license_name")
    flickr.add_argument("--hash-images", action="store_true")

    coco = subparsers.add_parser("prepare-coco", help="normalize the MS COCO Karpathy split")
    coco.add_argument("--karpathy-json", required=True)
    coco.add_argument("--dataset-root", required=True)
    coco.add_argument("--manifest-output", required=True)
    coco.add_argument("--source", required=True)
    coco.add_argument("--license", required=True, dest="license_name")
    coco.add_argument("--hash-images", action="store_true")

    sugar = subparsers.add_parser(
        "prepare-sugarcrepe", help="normalize the seven official SugarCrepe categories"
    )
    sugar.add_argument("--official-data-dir", required=True)
    sugar.add_argument("--dataset-root", required=True)
    sugar.add_argument("--manifest-output", required=True)
    sugar.add_argument("--source", required=True)
    sugar.add_argument("--license", required=True, dest="license_name")
    sugar.add_argument("--hash-images", action="store_true")

    finalize = subparsers.add_parser("finalize-run", help="finalize a non-reportable run")
    finalize.add_argument("run_dir")
    finalize.add_argument("--metrics", required=True)
    finalize.add_argument("--status", choices=sorted(FINALIZABLE_STATUSES), default="complete")
    finalize.add_argument("--notes")

    promote = subparsers.add_parser(
        "promote-run", help="promote one reviewed complete run to reportable"
    )
    promote.add_argument("run_dir")
    promote.add_argument("--reviewed-by", required=True)
    promote.add_argument("--notes")

    verify = subparsers.add_parser("verify-dataset", help="verify files declared by a manifest")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--root", required=True)

    table = subparsers.add_parser("build-table", help="build a Markdown table from valid runs")
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

        if args.command == "run-baseline":
            config = load_config(args.config)
            run_dir = create_run(
                config,
                config_path=args.config,
                output_root=args.output_root,
                run_id=args.run_id,
            )
            try:
                result = evaluate_baseline(config, use_cache=not args.no_cache)
                write_baseline_outputs(
                    run_dir,
                    result,
                    save_predictions=bool(config["evaluation"].get("save_predictions", True)),
                )
                finalize_run(
                    run_dir,
                    result.metrics,
                    status="complete",
                    notes="zero-shot baseline evaluation completed",
                )
                (Path(run_dir) / "metrics.pending.json").unlink(missing_ok=True)
            except Exception as exc:
                fail_run(run_dir, exc, notes="zero-shot baseline evaluation failed")
                raise
            print(run_dir)
            return 0

        if args.command == "prepare-flickr30k":
            manifest = prepare_flickr30k(
                args.karpathy_json,
                args.dataset_root,
                manifest_output=args.manifest_output,
                source=args.source,
                license_name=args.license_name,
                hash_images=args.hash_images,
            )
            print(f"prepared Flickr30K: {manifest['splits']}")
            print(args.manifest_output)
            return 0

        if args.command == "prepare-coco":
            manifest = prepare_coco(
                args.karpathy_json,
                args.dataset_root,
                manifest_output=args.manifest_output,
                source=args.source,
                license_name=args.license_name,
                hash_images=args.hash_images,
            )
            print(f"prepared MS COCO: {manifest['splits']}")
            print(args.manifest_output)
            return 0

        if args.command == "prepare-sugarcrepe":
            manifest = prepare_sugarcrepe(
                args.official_data_dir,
                args.dataset_root,
                manifest_output=args.manifest_output,
                source=args.source,
                license_name=args.license_name,
                hash_images=args.hash_images,
            )
            print(f"prepared SugarCrepe: {manifest['splits']}")
            print(args.manifest_output)
            return 0

        if args.command == "finalize-run":
            with Path(args.metrics).open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            finalize_run(args.run_dir, metrics, status=args.status, notes=args.notes)
            print(Path(args.run_dir) / "run.json")
            return 0

        if args.command == "promote-run":
            promote_run(args.run_dir, reviewed_by=args.reviewed_by, notes=args.notes)
            print(Path(args.run_dir) / "run.json")
            return 0

        if args.command == "verify-dataset":
            manifest = load_dataset_manifest(args.manifest)
            failures = verify_dataset(args.root, manifest)
            if failures:
                print("\n".join(failures))
                return 1
            if not manifest.get("files"):
                print("manifest does not declare files")
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
    except (
        ConfigError,
        ManifestError,
        SchemaValidationError,
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}")
        return 2
    return 2
