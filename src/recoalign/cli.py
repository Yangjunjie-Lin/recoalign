"""Command-line interface for reproducible research infrastructure."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from recoalign.analysis.results import collect_runs, render_markdown_table
from recoalign.config import ConfigError, config_digest, load_config
from recoalign.data.compositional_preparation import (
    prepare_aro,
    prepare_bivlc,
    prepare_winoground,
)
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
from recoalign.research_registry import RegistryError
from recoalign.schema_validation import SchemaValidationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recoalign")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="validate an experiment YAML file")
    validate.add_argument("config")

    subparsers.add_parser(
        "validate-research", help="validate hypothesis/experiment registries and bindings"
    )
    subparsers.add_parser("list-experiments", help="list registered scientific experiments")

    pivot_validate = subparsers.add_parser(
        "validate-pivot", help="preflight and freeze the PH001 A1/A2/A3 behavioral validation"
    )
    pivot_validate.add_argument("--config", default="research/pivot_validation/config.yaml")
    pivot_run = subparsers.add_parser(
        "run-pivot-validation", help="run the preregistered PH001 pilot or five-seed final"
    )
    pivot_run.add_argument("--config", default="research/pivot_validation/config.yaml")
    pivot_run.add_argument("--stage", choices=("pilot", "final"), required=True)

    causal_preregister = subparsers.add_parser(
        "preregister-causal-separation",
        help="freeze PIVOT_EXP_A2 hypotheses and pre-inference power analysis",
    )
    causal_preregister.add_argument("--study", choices=("PIVOT_EXP_A2",), required=True)
    causal_preregister.add_argument(
        "--config", default="research/causal_separation/PIVOT_EXP_A2/config.yaml"
    )
    causal_validate = subparsers.add_parser(
        "validate-causal-separation",
        help="validate and freeze the PIVOT_EXP_A2 design without loading weights",
    )
    causal_validate.add_argument("--study", choices=("PIVOT_EXP_A2",), required=True)
    causal_validate.add_argument(
        "--config", default="research/causal_separation/PIVOT_EXP_A2/config.yaml"
    )
    causal_validate.add_argument("--preflight-only", action="store_true")
    causal_run = subparsers.add_parser(
        "run-causal-separation", help="run the frozen PIVOT_EXP_A2 LLaVA experiment"
    )
    causal_run.add_argument("--study", choices=("PIVOT_EXP_A2",), required=True)
    causal_run.add_argument("--model", choices=("llava_1_5_7b",), required=True)
    causal_run.add_argument(
        "--config", default="research/causal_separation/PIVOT_EXP_A2/config.yaml"
    )
    causal_adjudicate = subparsers.add_parser(
        "adjudicate-causal-separation",
        help="apply the frozen PIVOT_EXP_A2 decision policy",
    )
    causal_adjudicate.add_argument("--study", choices=("PIVOT_EXP_A2",), required=True)
    causal_adjudicate.add_argument(
        "--config", default="research/causal_separation/PIVOT_EXP_A2/config.yaml"
    )

    construct_preregister = subparsers.add_parser(
        "preregister-construct-validity",
        help="freeze PIVOT_EXP_A3 construct hypotheses, contracts, and power analysis",
    )
    construct_preregister.add_argument("--study", choices=("PIVOT_EXP_A3",), required=True)
    construct_preregister.add_argument(
        "--config", default="research/construct_validity/PIVOT_EXP_A3/config.yaml"
    )
    construct_contract = subparsers.add_parser(
        "validate-answer-contract",
        help="validate the frozen PIVOT_EXP_A3 parser and tokenizer contract without weights",
    )
    construct_contract.add_argument("--study", choices=("PIVOT_EXP_A3",), required=True)
    construct_contract.add_argument(
        "--config", default="research/construct_validity/PIVOT_EXP_A3/config.yaml"
    )
    construct_validate = subparsers.add_parser(
        "validate-construct-validity",
        help="materialize and freeze PIVOT_EXP_A3 validation trials without VLM inference",
    )
    construct_validate.add_argument("--study", choices=("PIVOT_EXP_A3",), required=True)
    construct_validate.add_argument(
        "--config", default="research/construct_validity/PIVOT_EXP_A3/config.yaml"
    )
    construct_validate.add_argument("--preflight-only", action="store_true")
    construct_run = subparsers.add_parser(
        "run-construct-validity", help="run the frozen PIVOT_EXP_A3 LLaVA measurement"
    )
    construct_run.add_argument("--study", choices=("PIVOT_EXP_A3",), required=True)
    construct_run.add_argument("--model", choices=("llava_1_5_7b",), required=True)
    construct_run.add_argument(
        "--config", default="research/construct_validity/PIVOT_EXP_A3/config.yaml"
    )
    construct_adjudicate = subparsers.add_parser(
        "adjudicate-construct-validity",
        help="apply the frozen PIVOT_EXP_A3 task-specific decision policy",
    )
    construct_adjudicate.add_argument("--study", choices=("PIVOT_EXP_A3",), required=True)
    construct_adjudicate.add_argument(
        "--config", default="research/construct_validity/PIVOT_EXP_A3/config.yaml"
    )

    governed_run = subparsers.add_parser(
        "run-experiment", help="run a registered experiment through the governance lifecycle"
    )
    governed_run.add_argument("experiment_id", nargs="?")
    governed_run.add_argument("--experiment", dest="experiment_option")
    governed_run.add_argument("--model")
    governed_run.add_argument("--split")
    governed_run.add_argument("--batch-size", type=int)
    governed_run.add_argument("--no-cache", action="store_true")
    governed_run.add_argument("--config")
    governed_run.add_argument("--output-root", default="runs")
    governed_run.add_argument("--run-id")
    governed_run.add_argument("--seed", dest="seeds", type=int, action="append")
    governed_run.add_argument("--dry-run", action="store_true")

    vlm_evaluation = subparsers.add_parser(
        "run-vlm-eval", help="run one registered model x experiment evaluation cell"
    )
    vlm_evaluation.add_argument("--model", required=True)
    vlm_evaluation.add_argument("--experiment", required=True)
    vlm_evaluation.add_argument("--split")
    vlm_evaluation.add_argument("--seed", dest="seeds", type=int, action="append")
    vlm_evaluation.add_argument("--output")
    vlm_evaluation.add_argument("--batch-size", type=int)
    vlm_evaluation.add_argument("--no-cache", action="store_true")
    vlm_evaluation.add_argument("--dry-run", action="store_true")

    subparsers.add_parser("list-vlm-models", help="list registered Phase-2.1 VLM backbones")

    diagnosis = subparsers.add_parser(
        "run-interface-diagnosis",
        help="run EXP004 three-stage structured-interface diagnosis",
    )
    diagnosis.add_argument("--model", default="reference")
    diagnosis.add_argument("--config", default="configs/diagnosis/interface_diagnosis.yaml")
    diagnosis.add_argument("--output")
    diagnosis.add_argument("--seed", dest="seeds", type=int, action="append")
    diagnosis.add_argument("--dry-run", action="store_true")

    recoalign_toy = subparsers.add_parser(
        "train-recoalign-toy",
        help="run the CPU-friendly ReCoAlign interface training sanity check",
    )
    recoalign_toy.add_argument("--config", default="configs/models/recoalign.yaml")
    recoalign_toy.add_argument("--output", default="outputs/recoalign_toy")
    recoalign_toy.add_argument("--steps", type=int)
    recoalign_toy.add_argument("--seed", type=int, default=0)
    recoalign_toy.add_argument("--overfit-check", action="store_true")

    recoalign_training = subparsers.add_parser(
        "train-recoalign",
        help="run one audited ReCoAlign training stage on the configured data boundary",
    )
    recoalign_training.add_argument("--config", required=True)
    recoalign_training.add_argument("--output")
    recoalign_training.add_argument("--resume")
    recoalign_training.add_argument("--stop-after-epoch", type=int)
    recoalign_training.add_argument("--toy-data", action="store_true")
    recoalign_training.add_argument("--no-environment", action="store_true")

    recoalign_ablation = subparsers.add_parser(
        "run-recoalign-ablation", help="execute one config-driven ReCoAlign toy ablation"
    )
    recoalign_ablation.add_argument("--config", required=True)
    recoalign_ablation.add_argument("--output", required=True)

    baseline_fairness = subparsers.add_parser(
        "validate-training-fairness", help="validate controlled ReCoAlign baseline fields"
    )
    baseline_fairness.add_argument("--config", default="configs/training/baseline_matrix.yaml")

    training_sanity = subparsers.add_parser(
        "run-training-sanity", help="run controlled 100-sample training sanity checks"
    )
    training_sanity.add_argument("--config", default="configs/training/recoalign_stage1.yaml")
    training_sanity.add_argument("--output", default="outputs/TRAIN001_sanity")
    training_sanity.add_argument("--capture-environment", action="store_true")

    subparsers.add_parser(
        "validate-training-registry", help="validate TRAIN001-TRAIN003 and no-oracle contracts"
    )

    benchmark_validate = subparsers.add_parser(
        "validate-benchmark-matrix", help="validate the complete Phase-4 evaluation matrix"
    )
    benchmark_validate.add_argument(
        "--config", default="configs/benchmarks/comprehensive_matrix.yaml"
    )
    ablation_validate = subparsers.add_parser(
        "validate-benchmark-ablations", help="validate architecture/training/data ablation matrix"
    )
    ablation_validate.add_argument("--config", default="configs/benchmarks/ablation_matrix.yaml")
    benchmark_materialize = subparsers.add_parser(
        "materialize-benchmark-matrix", help="write every frozen Phase-4 cell config"
    )
    benchmark_materialize.add_argument(
        "--config", default="configs/benchmarks/comprehensive_matrix.yaml"
    )
    benchmark_materialize.add_argument("--output", required=True)
    benchmark_run = subparsers.add_parser(
        "run-vlm-benchmark", help="run or dry-run one unified BaseVLM benchmark cell"
    )
    benchmark_run.add_argument("--config", required=True)
    benchmark_run.add_argument("--output")
    benchmark_run.add_argument("--dry-run", action="store_true")
    benchmark_run.add_argument("--no-environment", action="store_true")
    benchmark_report = subparsers.add_parser(
        "build-comprehensive-report", help="build complete tables, figures, and decision report"
    )
    benchmark_report.add_argument(
        "--matrix-config", default="configs/benchmarks/comprehensive_matrix.yaml"
    )
    benchmark_report.add_argument("--results-root", default="outputs/comprehensive")
    benchmark_report.add_argument("--output-root", default="reports")

    mechanistic_registry = subparsers.add_parser(
        "validate-mechanistic-registry", help="validate registered Phase-5 ablations"
    )
    mechanistic_registry.add_argument(
        "--config", default="configs/ablations/mechanistic_registry.yaml"
    )
    mechanistic_run = subparsers.add_parser(
        "run-mechanistic-toy", help="run causal/intervention/representation toy analysis"
    )
    mechanistic_run.add_argument("--output", default="reports/mechanistic")
    mechanistic_run.add_argument("--seed", dest="seeds", type=int, action="append")
    mechanistic_report = subparsers.add_parser(
        "build-mechanistic-report", help="build ablation/mechanism/visualization/LaTeX outputs"
    )
    mechanistic_report.add_argument("--suite", default="reports/mechanistic/mechanistic_suite.json")
    mechanistic_report.add_argument("--output-root", default="reports")

    paper_package = subparsers.add_parser(
        "build-paper-package",
        help="audit and build conservative submission-preparation artifacts",
    )
    paper_package.add_argument("--root")
    paper_validate = subparsers.add_parser(
        "validate-paper-package",
        help="validate paper artifacts without requiring scientific GO",
    )
    paper_validate.add_argument("--root")
    paper_freeze = subparsers.add_parser(
        "freeze-paper-registry",
        help="write immutable protocol/config/checkpoint inventory without tagging",
    )
    paper_freeze.add_argument("--root")
    paper_tag = subparsers.add_parser(
        "create-paper-tag",
        help="create v2.0-paper-ready only after all scientific and Git gates pass",
    )
    paper_tag.add_argument("--root")
    paper_tag.add_argument("--tag", default="v2.0-paper-ready")
    paper_tag.add_argument("--create", action="store_true")

    decide = subparsers.add_parser(
        "decide-experiment",
        help="record a GO/NO-GO/INCONCLUSIVE YAML report from a result JSON",
    )
    decide.add_argument("result")
    decide.add_argument("--output")

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

    graph_vs_text = subparsers.add_parser(
        "run-graph-vs-text", help="run the complete multi-seed EXP001 controlled validation"
    )
    graph_vs_text.add_argument("--config", default="configs/graph_vs_text.yaml")
    graph_vs_text.add_argument("--output")

    graph_ablation = subparsers.add_parser(
        "run-graph-ablation", help="run Experiment B graph completeness/corruption ablations"
    )
    graph_ablation.add_argument("--config", default="configs/graph_ablation.yaml")
    graph_ablation.add_argument("--output")

    ood_composition = subparsers.add_parser(
        "run-ood-composition", help="run Experiment C on unseen compositions"
    )
    ood_composition.add_argument("--config", default="configs/ood_composition.yaml")

    benchmark = subparsers.add_parser(
        "run-structured-benchmark", help="run the complete Phase-1 mechanism suite"
    )
    benchmark.add_argument("--config-dir", default="configs")

    synthetic_evaluation = subparsers.add_parser(
        "evaluate-synthetic",
        help="evaluate the controlled synthetic world with multi-seed statistics",
    )
    synthetic_evaluation.add_argument("--config", default="configs/synthetic_benchmark.yaml")
    synthetic_evaluation.add_argument("--output")

    synthetic_generation = subparsers.add_parser(
        "generate-synthetic",
        help="materialize and validate a reconstructable synthetic dataset",
    )
    synthetic_generation.add_argument("--config", default="configs/synthetic_benchmark.yaml")
    synthetic_generation.add_argument("--count", type=int, default=1000)
    synthetic_generation.add_argument("--output", default="outputs/synthetic_world_example")
    synthetic_generation.add_argument("--seed", type=int)

    flickr = subparsers.add_parser(
        "prepare-flickr30k", help="normalize the Karpathy Flickr30K split"
    )
    _add_standard_preparation_arguments(flickr)
    flickr.add_argument("--karpathy-json", required=True)

    coco = subparsers.add_parser("prepare-coco", help="normalize the MS COCO Karpathy split")
    _add_standard_preparation_arguments(coco)
    coco.add_argument("--karpathy-json", required=True)

    sugar = subparsers.add_parser(
        "prepare-sugarcrepe", help="normalize the seven official SugarCrepe categories"
    )
    _add_standard_preparation_arguments(sugar)
    sugar.add_argument("--official-data-dir", required=True)

    aro = subparsers.add_parser(
        "prepare-aro", help="normalize a path-based export of the four ARO subsets"
    )
    _add_standard_preparation_arguments(aro)
    aro.add_argument("--source-jsonl", required=True)

    winoground = subparsers.add_parser(
        "prepare-winoground", help="normalize a path-based Winoground export"
    )
    _add_standard_preparation_arguments(winoground)
    winoground.add_argument("--source-jsonl", required=True)
    winoground.add_argument("--source-revision")
    winoground.add_argument("--exporter-version")
    winoground.add_argument("--downloaded-at")

    bivlc = subparsers.add_parser("prepare-bivlc", help="normalize a path-based BiVLC export")
    _add_standard_preparation_arguments(bivlc)
    bivlc.add_argument("--source-jsonl", required=True)

    finalize = subparsers.add_parser("finalize-run", help="finalize a non-reportable run")
    finalize.add_argument("run_dir")
    finalize.add_argument("--metrics", required=True)
    finalize.add_argument("--status", choices=sorted(FINALIZABLE_STATUSES), default="complete")
    finalize.add_argument("--notes")

    promote = subparsers.add_parser(
        "promote-run", help="promote one reviewed complete run to reportable"
    )
    promote.add_argument("run_dir")
    promote.add_argument(
        "--verification-run",
        help=(
            "required for Winoground reportable promotion; complete cache-disabled rerun "
            "with identical provenance"
        ),
    )
    promote.add_argument(
        "--prediction-review",
        help=(
            "required for Winoground reportable promotion; CSV covering every prediction "
            "sample ID and mapping review"
        ),
    )
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
    arguments = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(arguments)
    command_line = shlex.join(["recoalign", *arguments])
    try:
        if args.command == "validate-config":
            structured = False
            method_config = False
            try:
                config = load_config(args.config)
            except ConfigError as baseline_error:
                try:
                    from experiments.runtime import load_config as load_structured_config

                    config = load_structured_config(args.config)
                    structured = True
                except (FileNotFoundError, OSError, TypeError, ValueError):
                    try:
                        raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
                        if not isinstance(raw, dict):
                            raise ValueError("configuration root must be a mapping")
                        if "benchmark" in raw and "generation" in raw:
                            from recoalign.evaluation.vlm_benchmark.runner import (
                                validate_benchmark_config,
                            )

                            validate_benchmark_config(raw)
                            config = raw
                        elif "benchmarks" in raw and "models" in raw:
                            from recoalign.evaluation.vlm_benchmark.matrix import (
                                validate_evaluation_matrix,
                            )

                            validate_evaluation_matrix(raw)
                            config = raw
                        elif {"benchmark", "model", "evaluation"}.issubset(raw):
                            _validate_synthetic_benchmark_config(raw)
                            config = raw
                        elif "stage" in raw:
                            from recoalign.training.trainer import load_training_config

                            config = load_training_config(args.config).raw
                        elif "ablation" in raw:
                            from recoalign.training.ablations import load_ablation_config
                            from recoalign.training.trainer import TrainingConfig

                            config = load_ablation_config(args.config)
                            TrainingConfig.from_mapping(config)
                        else:
                            from recoalign.training.recoalign_toy import load_recoalign_config

                            config = load_recoalign_config(args.config).to_dict()
                        method_config = True
                    except (FileNotFoundError, OSError, TypeError, ValueError) as method_error:
                        raise baseline_error from method_error
            print(f"valid: {args.config}")
            if structured or method_config:
                canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
                digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            else:
                digest = config_digest(config)
            print(f"sha256: {digest}")
            return 0

        if args.command == "validate-research":
            from recoalign.research_registry import validate_research_registries

            print(json.dumps(validate_research_registries(), indent=2, sort_keys=True))
            return 0

        if args.command == "validate-pivot":
            from recoalign.pivot_validation import preflight_pivot_validation

            report = preflight_pivot_validation(args.config)
            print(
                json.dumps(
                    {"status": report["status"], "weights_loaded": report["weights_loaded"]},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "run-pivot-validation":
            from recoalign.pivot_validation import run_pivot_validation

            report = run_pivot_validation(args.config, stage=args.stage)
            print(
                json.dumps(
                    {
                        "stage": args.stage,
                        "decision": report["metrics"]["mechanism"]["decision"],
                        "classification": report["metrics"]["mechanism"]["classification"],
                        "prediction_count": report["metrics"]["prediction_count"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "preregister-causal-separation":
            from recoalign.causal_separation import preregister_causal_separation

            report = preregister_causal_separation(args.config)
            print(
                json.dumps(
                    {
                        "study": args.study,
                        "status": report["status"],
                        "power": report["power"]["status"],
                        "weights_loaded": report["weights_loaded"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "validate-causal-separation":
            from recoalign.causal_separation import validate_causal_separation

            report = validate_causal_separation(
                args.config, preflight_only=bool(args.preflight_only)
            )
            print(
                json.dumps(
                    {
                        "study": args.study,
                        "status": report["status"],
                        "weights_loaded": report["weights_loaded"],
                        "inference_started": report["inference_started"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "run-causal-separation":
            from recoalign.causal_separation import run_causal_separation

            report = run_causal_separation(args.config, model_name=args.model)
            print(
                json.dumps(
                    {
                        "study": args.study,
                        "status": report["status"],
                        "prediction_count": report["prediction_count"],
                        "adjudication_pending": report["adjudication_pending"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "adjudicate-causal-separation":
            from recoalign.causal_separation import adjudicate_causal_separation

            report = adjudicate_causal_separation(args.config)
            print(
                json.dumps(
                    {
                        "study": args.study,
                        "outcome": report["decision"]["outcome"],
                        "authorization": report["decision"]["authorization"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "preregister-construct-validity":
            from recoalign.construct_validity import preregister_construct_validity

            report = preregister_construct_validity(args.config)
            print(
                json.dumps(
                    {
                        "study": args.study,
                        "status": report["status"],
                        "power": report["power"]["status"],
                        "weights_loaded": report["weights_loaded"],
                        "inference_started": report["inference_started"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "validate-answer-contract":
            from recoalign.construct_validity import validate_answer_contract

            report = validate_answer_contract(args.config)
            print(
                json.dumps(
                    {
                        "study": args.study,
                        "passed": report["passed"],
                        "parser": report["parser"]["passed"],
                        "tokenizer": report["tokenizer"]["passed"],
                        "weights_loaded": report["weights_loaded"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "validate-construct-validity":
            from recoalign.construct_validity import validate_construct_validity

            report = validate_construct_validity(
                args.config, preflight_only=bool(args.preflight_only)
            )
            print(
                json.dumps(
                    {
                        "study": args.study,
                        "status": report["status"],
                        "passed": report["passed"],
                        "weights_loaded": report["weights_loaded"],
                        "inference_started": report["inference_started"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "run-construct-validity":
            from recoalign.construct_validity import run_construct_validity

            report = run_construct_validity(args.config, model_name=args.model)
            print(
                json.dumps(
                    {
                        "study": args.study,
                        "status": report["status"],
                        "prediction_count": report["prediction_count"],
                        "adjudication_pending": report["adjudication_pending"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "adjudicate-construct-validity":
            from recoalign.construct_validity import adjudicate_construct_validity_run

            report = adjudicate_construct_validity_run(args.config)
            print(
                json.dumps(
                    {
                        "study": args.study,
                        "outcome": report["decision"]["outcome"],
                        "authorization": report["decision"]["authorization"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "list-experiments":
            from recoalign.research_registry import load_experiment_registry

            for experiment in load_experiment_registry()["experiments"]:
                print(
                    f"{experiment['experiment_id']}\t{experiment['hypothesis_id']}\t"
                    f"{experiment['title']}\t{experiment['status']}"
                )
            return 0

        if args.command == "run-experiment":
            experiment_id = args.experiment_option or args.experiment_id
            if not experiment_id:
                raise ValueError("run-experiment requires EXPxxx or --experiment EXPxxx")
            if experiment_id == "EXP004":
                if args.config and args.model:
                    raise ValueError(
                        "EXP004 uses the registered diagnosis config; use run-interface-diagnosis "
                        "for a custom diagnostic config"
                    )
                from diagnosis.interface_diagnosis.runner import run_interface_diagnosis

                run_dir = run_interface_diagnosis(
                    model_name=args.model or "reference",
                    seeds=args.seeds,
                    dry_run=args.dry_run,
                )
            elif args.model:
                if args.config:
                    raise ValueError(
                        "model-matrix runs use the registered experiment config; "
                        "--config is not allowed"
                    )
                from recoalign.vlm_evaluation import run_vlm_evaluation

                run_dir = run_vlm_evaluation(
                    model_name=args.model,
                    experiment_id=experiment_id,
                    split=args.split,
                    seeds=args.seeds,
                    dry_run=args.dry_run,
                    cache_enabled=not args.no_cache,
                    batch_size=args.batch_size,
                )
            else:
                from recoalign.governance import run_registered_experiment

                run_dir = run_registered_experiment(
                    experiment_id,
                    config_path=args.config,
                    output_root=args.output_root,
                    run_id=args.run_id,
                    dry_run=args.dry_run,
                    seeds=args.seeds,
                    command=command_line,
                )
            print(run_dir)
            return 0

        if args.command == "run-vlm-eval":
            from recoalign.vlm_evaluation import run_vlm_evaluation

            output = run_vlm_evaluation(
                model_name=args.model,
                experiment_id=args.experiment,
                split=args.split,
                seeds=args.seeds,
                output_dir=args.output,
                dry_run=args.dry_run,
                cache_enabled=not args.no_cache,
                batch_size=args.batch_size,
            )
            print(output)
            return 0

        if args.command == "list-vlm-models":
            from recoalign.models.vlm.registry import get_model_registry

            registry = get_model_registry()
            for name in registry.names():
                definition = registry.definition(name)
                print(
                    f"{name}\tadapter={definition.payload['adapter']}\t"
                    f"scientific_evidence={str(definition.scientific_evidence).lower()}"
                )
            return 0

        if args.command == "run-interface-diagnosis":
            from diagnosis.interface_diagnosis.runner import run_interface_diagnosis

            output = run_interface_diagnosis(
                model_name=args.model,
                config_path=args.config,
                output_dir=args.output,
                seeds=args.seeds,
                dry_run=args.dry_run,
            )
            print(output)
            return 0

        if args.command == "train-recoalign-toy":
            from recoalign.training.recoalign_toy import (
                load_recoalign_config,
                overfit_sanity_test,
                run_toy_training,
            )

            config = load_recoalign_config(args.config)
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = output_dir / "checkpoint.pt"
            result = run_toy_training(
                config,
                steps=args.steps or 30,
                seed=args.seed,
                checkpoint_path=checkpoint,
            )
            payload = {
                "method": "recoalign",
                "config": config.to_dict(),
                "steps": result["steps"],
                "seed": result["seed"],
                "initial_loss": result["initial_loss"],
                "final_loss": result["final_loss"],
                "loss_decreased": result["loss_decreased"],
                "loss_history": result["loss_history"],
                "checkpoint": result["checkpoint"],
            }
            if args.overfit_check:
                payload["overfit_sanity"] = overfit_sanity_test(config, seed=args.seed)
            atomic_write_json(output_dir / "metrics.json", payload)
            (output_dir / "config.resolved.yaml").write_text(
                yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
            )
            print(output_dir)
            return 0

        if args.command == "train-recoalign":
            if not args.toy_data:
                raise ValueError(
                    "this repository build exposes only the audited --toy-data loader; "
                    "real dataset loaders must be registered before training"
                )
            from recoalign.training.recoalign_toy import make_toy_batch
            from recoalign.training.trainer import ReCoAlignTrainer, load_training_config

            training_config = load_training_config(args.config)
            trainer = ReCoAlignTrainer.from_config(
                training_config,
                run_dir=args.output,
                capture_environment_metadata=not args.no_environment,
            )
            train_batch = make_toy_batch(
                batch_size=training_config.batch_size,
                config=trainer.model.config,
                seed=training_config.seed + 1,
            ).as_dict()
            validation_batch = make_toy_batch(
                batch_size=training_config.batch_size,
                config=trainer.model.config,
                seed=training_config.seed + 2,
            ).as_dict()
            result = trainer.fit(
                [train_batch],
                [validation_batch],
                resume_from=args.resume,
                stop_after_epoch=args.stop_after_epoch,
            )
            print(trainer.run_dir)
            print(f"loss_decreased={str(result['loss_decreased']).lower()}")
            return 0

        if args.command == "run-recoalign-ablation":
            from recoalign.training.ablations import load_ablation_config, run_toy_ablation

            result = run_toy_ablation(
                load_ablation_config(args.config),
                run_dir=args.output,
            )
            print(args.output)
            print(f"loss_decreased={str(result['loss_decreased']).lower()}")
            return 0

        if args.command == "validate-training-fairness":
            from recoalign.training.fairness import validate_baseline_fairness

            payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("baselines"), dict):
                raise ValueError("baseline fairness config requires a baselines mapping")
            print(
                json.dumps(
                    validate_baseline_fairness(payload["baselines"]),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "run-training-sanity":
            from recoalign.training.sanity import run_training_sanity_suite
            from recoalign.training.trainer import load_training_config

            config = load_training_config(args.config)
            result = run_training_sanity_suite(
                config.raw,
                output_dir=args.output,
                capture_environment_metadata=args.capture_environment,
            )
            print(args.output)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0

        if args.command == "validate-training-registry":
            from recoalign.training.registry import validate_training_registry

            print(json.dumps(validate_training_registry(), indent=2, sort_keys=True))
            return 0

        if args.command == "validate-benchmark-matrix":
            from recoalign.evaluation.vlm_benchmark.matrix import (
                load_matrix_config,
                validate_evaluation_matrix,
            )

            print(
                json.dumps(
                    validate_evaluation_matrix(load_matrix_config(args.config)),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "validate-benchmark-ablations":
            from recoalign.evaluation.vlm_benchmark.matrix import (
                load_ablation_matrix,
                validate_evaluation_matrix,
            )

            resolved = load_ablation_matrix(args.config)
            print(
                json.dumps(
                    {
                        "valid": True,
                        "architecture": resolved["definition"]["architecture"],
                        "training": resolved["definition"]["training"],
                        "data": resolved["definition"]["data"],
                        "matrix": validate_evaluation_matrix(resolved["resolved_matrix"]),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "materialize-benchmark-matrix":
            from recoalign.evaluation.vlm_benchmark.matrix import (
                load_matrix_config,
                materialize_matrix_configs,
            )

            manifest = materialize_matrix_configs(load_matrix_config(args.config), args.output)
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0

        if args.command == "run-vlm-benchmark":
            from recoalign.evaluation.vlm_benchmark.runner import run_benchmark_cell

            output = run_benchmark_cell(
                args.config,
                output_dir=args.output,
                dry_run=args.dry_run,
                capture_environment_metadata=not args.no_environment,
            )
            print(output)
            return 0

        if args.command == "build-comprehensive-report":
            from recoalign.evaluation.vlm_benchmark.reporting import generate_comprehensive_reports

            report = generate_comprehensive_reports(
                matrix_config=args.matrix_config,
                results_root=args.results_root,
                output_root=args.output_root,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0

        if args.command == "validate-mechanistic-registry":
            from recoalign.analysis.mechanistic.registry import validate_mechanistic_registry

            print(json.dumps(validate_mechanistic_registry(args.config), indent=2, sort_keys=True))
            return 0

        if args.command == "run-mechanistic-toy":
            from recoalign.analysis.mechanistic.runner import run_toy_mechanistic_suite

            seeds = tuple(args.seeds) if args.seeds else (101, 202, 303)
            result = run_toy_mechanistic_suite(output_dir=args.output, seeds=seeds)
            print(args.output)
            print(
                json.dumps(
                    {
                        "scientific_status": result["scientific_status"],
                        "registered_ablation_count": result["registered_ablation_count"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "build-mechanistic-report":
            from recoalign.analysis.mechanistic.reporting import generate_mechanistic_reports

            report = generate_mechanistic_reports(args.suite, output_root=args.output_root)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0

        if args.command == "build-paper-package":
            from recoalign.paper_package import build_paper_package

            result = build_paper_package(args.root)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0

        if args.command == "validate-paper-package":
            from recoalign.paper_package import validate_paper_package

            result = validate_paper_package(args.root)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0

        if args.command == "freeze-paper-registry":
            from recoalign.paper_package.freeze import build_frozen_registry

            result = build_frozen_registry(args.root)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0

        if args.command == "create-paper-tag":
            from recoalign.paper_package import create_paper_tag

            result = create_paper_tag(args.root, tag=args.tag, create=args.create)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0

        if args.command == "decide-experiment":
            from recoalign.governance import decide_existing_result

            report = decide_existing_result(args.result, output=args.output)
            print(json.dumps(report, indent=2, sort_keys=True))
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

        if args.command == "run-graph-vs-text":
            from experiments.graph_vs_text.runner import run

            result = run(args.config, output_dir=args.output)
            print(args.output or "outputs/EXP001")
            print(f"decision={result['decision']}")
            return 0

        if args.command == "run-graph-ablation":
            from experiments.graph_ablation.runner import run

            result = run(args.config, output_dir=args.output)
            print(args.output or "outputs/EXP002")
            print(f"decision={result['decision']}")
            return 0

        if args.command == "run-ood-composition":
            from recoalign.governance import run_registered_experiment

            print(
                run_registered_experiment("EXP003", config_path=args.config, command=command_line)
            )
            return 0

        if args.command == "run-structured-benchmark":
            from recoalign.governance import run_registered_experiment

            config_dir = Path(args.config_dir)
            paths = [
                run_registered_experiment(
                    "EXP001",
                    config_path=config_dir / "graph_vs_text.yaml",
                    command=command_line,
                ),
                run_registered_experiment(
                    "EXP002",
                    config_path=config_dir / "graph_ablation.yaml",
                    command=command_line,
                ),
                run_registered_experiment(
                    "EXP003",
                    config_path=config_dir / "ood_composition.yaml",
                    command=command_line,
                ),
            ]
            print(json.dumps([str(path) for path in paths], indent=2))
            return 0

        if args.command == "evaluate-synthetic":
            from recoalign.synthetic_world.evaluation import evaluate_synthetic

            metrics = evaluate_synthetic(args.config, output_dir=args.output)
            destination = Path(
                args.output
                or yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["benchmark"][
                    "output_dir"
                ]
            )
            print(destination)
            print(json.dumps(metrics["conditions"], indent=2, sort_keys=True))
            return 0

        if args.command == "generate-synthetic":
            from recoalign.synthetic_world.evaluation import generate_example_dataset

            report = generate_example_dataset(
                args.config,
                count=args.count,
                output_dir=args.output,
                seed=args.seed,
            )
            print(args.output)
            print(json.dumps(report, indent=2, sort_keys=True))
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
            return _print_prepared("Flickr30K", manifest, args.manifest_output)

        if args.command == "prepare-coco":
            manifest = prepare_coco(
                args.karpathy_json,
                args.dataset_root,
                manifest_output=args.manifest_output,
                source=args.source,
                license_name=args.license_name,
                hash_images=args.hash_images,
            )
            return _print_prepared("MS COCO", manifest, args.manifest_output)

        if args.command == "prepare-sugarcrepe":
            manifest = prepare_sugarcrepe(
                args.official_data_dir,
                args.dataset_root,
                manifest_output=args.manifest_output,
                source=args.source,
                license_name=args.license_name,
                hash_images=args.hash_images,
            )
            return _print_prepared("SugarCrepe", manifest, args.manifest_output)

        if args.command == "prepare-aro":
            manifest = prepare_aro(
                args.source_jsonl,
                args.dataset_root,
                manifest_output=args.manifest_output,
                source=args.source,
                license_name=args.license_name,
                hash_images=args.hash_images,
            )
            return _print_prepared("ARO", manifest, args.manifest_output)

        if args.command == "prepare-winoground":
            manifest = prepare_winoground(
                args.source_jsonl,
                args.dataset_root,
                manifest_output=args.manifest_output,
                source=args.source,
                license_name=args.license_name,
                hash_images=args.hash_images,
                source_revision=args.source_revision,
                exporter_version=args.exporter_version,
                downloaded_at=args.downloaded_at,
            )
            return _print_prepared("Winoground", manifest, args.manifest_output)

        if args.command == "prepare-bivlc":
            manifest = prepare_bivlc(
                args.source_jsonl,
                args.dataset_root,
                manifest_output=args.manifest_output,
                source=args.source,
                license_name=args.license_name,
                hash_images=args.hash_images,
            )
            return _print_prepared("BiVLC", manifest, args.manifest_output)

        if args.command == "finalize-run":
            with Path(args.metrics).open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            finalize_run(args.run_dir, metrics, status=args.status, notes=args.notes)
            print(Path(args.run_dir) / "run.json")
            return 0

        if args.command == "promote-run":
            promote_run(
                args.run_dir,
                verification_run=args.verification_run,
                prediction_review=args.prediction_review,
                reviewed_by=args.reviewed_by,
                notes=args.notes,
            )
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
        RegistryError,
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


def _validate_synthetic_benchmark_config(config: dict[str, object]) -> None:
    """Validate the CPU/reference synthetic benchmark schema without importing Torch."""

    required_sections = ("benchmark", "model", "evaluation")
    for section in required_sections:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"synthetic benchmark requires a {section} mapping")

    benchmark = config["benchmark"]
    assert isinstance(benchmark, dict)
    for field in ("name", "version", "seed", "count", "output_dir"):
        if field not in benchmark:
            raise ValueError(f"synthetic benchmark.benchmark.{field} is required")
    if any(
        isinstance(benchmark[field], bool)
        or not isinstance(benchmark[field], (int, str))
        for field in ("seed", "count")
    ):
        raise ValueError("synthetic benchmark seed/count must be scalar values")
    if int(benchmark["count"]) <= 0:
        raise ValueError("synthetic benchmark count must be positive")

    model = config["model"]
    assert isinstance(model, dict)
    if not str(model.get("backend", "")).strip():
        raise ValueError("synthetic benchmark model.backend is required")

    evaluation = config["evaluation"]
    assert isinstance(evaluation, dict)
    seeds = evaluation.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("synthetic benchmark evaluation.seeds must be non-empty")
    minimum = 5 if bool(evaluation.get("critical", False)) else 3
    if len(seeds) < minimum:
        raise ValueError(f"synthetic benchmark requires at least {minimum} evaluation seeds")


def _add_standard_preparation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--license", required=True, dest="license_name")
    parser.add_argument("--hash-images", action="store_true")


def _print_prepared(name: str, manifest: dict[str, object], path: str) -> int:
    print(f"prepared {name}: {manifest['splits']}")
    print(path)
    return 0
