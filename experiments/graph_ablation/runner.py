"""Execute EXP002 graph-completeness and structural-necessity validation."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw

from datasets.records import SceneRecord
from evaluation.statistics import multiple_seed_summary
from experiments.graph_ablation.analysis import (
    FULL,
    IMAGE_ONLY,
    LABEL_RANDOMIZED,
    OBJECT_SWAP,
    ORDER_RANDOMIZED,
    RANDOM_GRAPH,
    RELATION_FLIP,
    RELATION_GROUPS,
    analyze_seed,
    partial_condition,
)
from experiments.runtime import (
    build_generator,
    build_vlm,
    checkpoint_provenance,
    git_commit,
    load_config,
)
from models.vlm.base import BaseVLM, PreparedInput
from recoalign.reproducibility import get_git_metadata, utc_now
from recoalign.research_registry import get_registered_experiment, validate_research_registries
from recoalign.schema_validation import repository_root, validate_payload
from recoalign.synthetic_world.corruption import (
    flip_relation,
    randomize_graph,
    remove_relation,
    swap_entity,
)
from recoalign.synthetic_world.ontology import RELATIONS, normalize_relation
from recoalign.synthetic_world.scene_graph import SceneGraph
from recoalign.synthetic_world.validation import validate_dataset
from research.decision_engine import evaluate_decision, render_decision_report

PADDING_CANDIDATES = (" <pad>", " neutral", " x", " .", " 0")


def run(
    config_path: str | Path = "configs/graph_ablation.yaml",
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run every registered seed and write the authoritative EXP002 artifact bundle."""

    validate_research_registries()
    config = deepcopy(load_config(config_path))
    _validate_exp002_config(config)
    seeds = [int(value) for value in config["experiment"]["seeds"]]
    minimum = 5 if bool(config["evaluation"].get("critical", True)) else 3
    if len(seeds) < minimum or len(set(seeds)) != len(seeds):
        raise ValueError(f"EXP002 requires at least {minimum} unique seeds")
    output = Path(output_dir or config["experiment"]["output_dir"])
    _ensure_new_bundle(output)
    resolved = deepcopy(config)
    resolved["experiment"]["output_dir"] = output.as_posix()
    resolved["meta"]["executed_config_path"] = str(Path(config_path).resolve().as_posix())
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8", newline="\n"
    )

    per_seed: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    combined_corruptions: list[dict[str, Any]] = []
    try:
        with (output / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as target:
            for seed in seeds:
                seed_dir = output / "seeds" / str(seed)
                metrics = run_seed_config(resolved, output_dir=seed_dir, seed=seed)
                per_seed.append({"seed": seed, "metrics": metrics})
                integrity_rows.append(_seed_integrity(seed_dir, seed))
                with (seed_dir / "predictions.jsonl").open(encoding="utf-8") as source:
                    for line in source:
                        target.write(line)
                payload = json.loads(
                    (seed_dir / "corruption_manifest.json").read_text(encoding="utf-8")
                )
                combined_corruptions.extend(payload["entries"])
    except Exception as exc:
        _record_failed_bundle(output, resolved, seeds, per_seed, integrity_rows, exc)
        raise

    registration = get_registered_experiment("EXP002")
    aggregate = _registered_aggregate(registration, per_seed, resolved)
    descriptive = _descriptive_aggregate(per_seed, resolved)
    result = {
        "schema_version": 1,
        "run_id": f"exp002-{utc_now().replace(':', '').replace('-', '')}",
        "experiment_id": "EXP002",
        "hypothesis": "H002",
        "model": f"frozen-{resolved['model']['backend']}",
        "dataset": f"{resolved['data']['dataset']}@{resolved['data']['version']}",
        "seed": seeds,
        "metrics": {"per_seed": per_seed, "aggregate": aggregate, **descriptive},
        "confidence_interval": {
            path: value["confidence_interval"] for path, value in aggregate.items()
        },
        "statistical_test": {
            path: value["statistical_test"] for path, value in aggregate.items()
        },
        "decision": "INCONCLUSIVE",
        "timestamp": utc_now(),
        "integrity": {
            "registry_validated": True,
            "dataset_version": registration["dataset"]["version"],
            "per_seed": integrity_rows,
        },
        "status": "complete",
    }
    report = evaluate_decision(registration, result)
    validate_payload("decision_report", report)
    result["decision"] = report["final_decision"]
    validate_payload("experiment_result", result)
    _write_json(output / "metrics.json", result)
    (output / "decision_report.yaml").write_text(
        yaml.safe_dump(report, sort_keys=False), encoding="utf-8", newline="\n"
    )
    (output / "decision_report.md").write_text(
        render_decision_report(report), encoding="utf-8", newline="\n"
    )
    _write_json(
        output / "corruption_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": "EXP002",
            "seeds": seeds,
            "entry_count": len(combined_corruptions),
            "operation_counts": dict(
                sorted(Counter(row["operation"] for row in combined_corruptions).items())
            ),
            "entries": combined_corruptions,
        },
    )
    _write_figures(output, descriptive, resolved)
    _write_relation_table(output, descriptive)
    _write_narrative_reports(output, result, report, resolved)

    run_record = {
        "schema_version": 1,
        "run_id": result["run_id"],
        "experiment_id": "EXP002",
        "hypothesis_id": "H002",
        "status": "complete",
        "decision": report["final_decision"],
        "completed_at": utc_now(),
        "model_backend": resolved["model"]["backend"],
        "checkpoint": checkpoint_provenance(resolved),
        "dataset": result["dataset"],
        "seeds": seeds,
        "scientific_interpretation": _scientific_interpretation(report, resolved),
        "git": get_git_metadata(repository_root()),
    }
    _write_json(output / "run.json", run_record)
    _write_json(output / "manifest.json", _bundle_manifest(output, run_record))
    return result


def run_seed_config(
    config: dict[str, Any],
    *,
    output_dir: str | Path,
    seed: int | None = None,
) -> dict[str, Any]:
    """Execute one paired seed; used by both the bundle and governance runners."""

    resolved = deepcopy(config)
    _validate_exp002_config(resolved)
    actual_seed = int(resolved["experiment"]["seed"] if seed is None else seed)
    resolved["experiment"]["seed"] = actual_seed
    resolved["model"]["seed"] = actual_seed
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    resolved["experiment"]["output_dir"] = output.as_posix()
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8", newline="\n"
    )

    generator = build_generator(resolved)
    synthetic = resolved["synthetic"]
    relation_order = tuple(str(value) for value in synthetic.get("relation_order", ()))
    records = generator.generate(
        int(synthetic.get("count", 120)),
        output_dir=output / "dataset",
        seed=actual_seed,
        split="test",
        relation_order=relation_order or None,
    )
    integrity = validate_dataset(
        records, check_images=bool(synthetic.get("write_images", True))
    )
    model = build_vlm(resolved)
    ratios = tuple(float(value) for value in resolved["corruption"]["partial_ratios"])
    corruption_ratio = float(resolved["corruption"].get("wrong_graph_ratio", 1.0))
    rows: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []

    for record in records:
        graph = SceneGraph(record.objects, record.relations)
        sample_seed = int(record.metadata.get("sample_seed", actual_seed))
        random_result = randomize_graph(graph, seed=sample_seed + 3001)
        full_input, random_input = _length_matched_pair(
            model, record, graph, random_result.corrupted
        )
        rows.append(_predict_row(model, record, FULL, full_input))
        rows.append(
            _predict_row(
                model,
                record,
                RANDOM_GRAPH,
                random_input,
                operation="random_graph",
                ratio=1.0,
            )
        )
        random_manifest = random_result.to_manifest()
        random_manifest["details"]["length_control"] = {
            "full_input_tokens": full_input.input_tokens,
            "random_input_tokens": random_input.input_tokens,
            "matched": full_input.input_tokens == random_input.input_tokens,
        }
        manifest_entries.append(_tag_manifest(random_manifest, record, actual_seed))

        image_input = model.prepare_input(record, IMAGE_ONLY)
        rows.append(_predict_row(model, record, IMAGE_ONLY, image_input))

        supporting = tuple(
            int(value) for value in record.metadata.get("query", {}).get("supporting_edges", ())
        )
        for offset, ratio in enumerate(ratios):
            mutation = remove_relation(
                graph,
                ratio=ratio,
                seed=sample_seed + 1001 + offset,
                critical_edge_indices=supporting,
            )
            condition = partial_condition(ratio)
            prepared = _prepare_graph_input(model, record, condition, mutation.corrupted)
            rows.append(
                _predict_row(
                    model,
                    record,
                    condition,
                    prepared,
                    operation="remove_relation",
                    ratio=ratio,
                )
            )
            manifest_entries.append(_tag_manifest(mutation.to_manifest(), record, actual_seed))

        for condition, mutation in (
            (
                RELATION_FLIP,
                flip_relation(graph, ratio=corruption_ratio, seed=sample_seed + 2001),
            ),
            (
                OBJECT_SWAP,
                swap_entity(graph, ratio=corruption_ratio, seed=sample_seed + 2002),
            ),
        ):
            prepared = _prepare_graph_input(model, record, condition, mutation.corrupted)
            rows.append(
                _predict_row(
                    model,
                    record,
                    condition,
                    prepared,
                    operation=mutation.operation,
                    ratio=corruption_ratio,
                )
            )
            manifest_entries.append(_tag_manifest(mutation.to_manifest(), record, actual_seed))

        order_edges = list(graph.edges)
        random.Random(sample_seed + 4001).shuffle(order_edges)
        if len(order_edges) > 1 and order_edges == list(graph.edges):
            order_edges.reverse()
        order_graph = SceneGraph(graph.nodes, tuple(order_edges))
        order_input = _prepare_graph_input(
            model,
            record,
            ORDER_RANDOMIZED,
            order_graph,
            serialization="canonical_random_order",
            semantic_hash=_fact_hash(record),
        )
        rows.append(
            _predict_row(
                model,
                record,
                ORDER_RANDOMIZED,
                order_input,
                operation="serialization_order_randomization",
                ratio=0.0,
            )
        )
        manifest_entries.append(
            _tag_manifest(
                _auxiliary_manifest(
                    graph,
                    order_graph.to_dict(),
                    operation="serialization_order_randomization",
                    seed=sample_seed + 4001,
                    details={"semantic_content_preserved": True},
                ),
                record,
                actual_seed,
            )
        )

        label_map = _opaque_label_map(sample_seed + 5001)
        label_input = _prepare_graph_input(
            model,
            record,
            LABEL_RANDOMIZED,
            graph,
            serialization="opaque_relation_labels",
            relation_labels=label_map,
        )
        rows.append(
            _predict_row(
                model,
                record,
                LABEL_RANDOMIZED,
                label_input,
                operation="relation_label_randomization",
                ratio=1.0,
            )
        )
        labeled_graph = graph.to_dict()
        labeled_graph["relations"] = [
            {**edge, "relation": label_map[edge["relation"]]}
            for edge in labeled_graph["relations"]
        ]
        manifest_entries.append(
            _tag_manifest(
                _auxiliary_manifest(
                    graph,
                    labeled_graph,
                    operation="relation_label_randomization",
                    seed=sample_seed + 5001,
                    details={"label_map": label_map, "legend_exposed_to_model": False},
                ),
                record,
                actual_seed,
            )
        )

    metrics = analyze_seed(
        rows,
        partial_ratios=ratios,
        bootstrap_samples=int(resolved["evaluation"]["bootstrap_samples"]),
        seed=int(resolved["evaluation"]["statistics_seed"]) + actual_seed,
    )
    leakage_passed = _no_control_metadata_in_prompts(rows)
    length_passed = bool(metrics["anti_shortcut"]["graph_length_matching"]["passed"])
    expected_entries = len(records) * (len(ratios) + 5)
    corruption_complete = len(manifest_entries) == expected_entries and all(
        entry["changed"] or entry["operation"] == "serialization_order_randomization"
        for entry in manifest_entries
    )
    metrics.update(
        {
            "experiment": "graph_completeness_structural_necessity",
            "model_backend": resolved["model"]["backend"],
            "seed": actual_seed,
            "dataset_size": len(records),
            "integrity": {
                **integrity,
                "corruption_manifest_complete": corruption_complete,
                "random_graph_length_matched": length_passed,
                "original_graph_not_in_prompt_metadata": leakage_passed,
            },
        }
    )
    _write_json(output / "metrics.json", metrics)
    with (output / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    _write_json(
        output / "corruption_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": "EXP002",
            "seed": actual_seed,
            "entry_count": len(manifest_entries),
            "expected_entry_count": expected_entries,
            "complete": corruption_complete,
            "operation_counts": dict(
                sorted(Counter(row["operation"] for row in manifest_entries).items())
            ),
            "entries": manifest_entries,
        },
    )
    seed_run = {
        "experiment": "EXP002",
        "status": "complete",
        "seed": actual_seed,
        "model_backend": resolved["model"]["backend"],
        "checkpoint": checkpoint_provenance(resolved),
        "dataset_size": len(records),
        "prediction_count": len(rows),
        "git_commit": git_commit(),
        "manifest_assertions": {
            "generator": "recoalign.synthetic_world.v2",
            "corruption_manifest_complete": corruption_complete,
            "random_graph_length_matched": length_passed,
            "original_graph_not_in_prompt_metadata": leakage_passed,
        },
    }
    _write_json(output / "run.json", seed_run)
    (output / "run.log").write_text(
        f"{utc_now()} RUN_COMPLETE experiment=EXP002 seed={actual_seed}\n",
        encoding="utf-8",
        newline="\n",
    )
    if not all(seed_run["manifest_assertions"].values()):
        raise RuntimeError(f"EXP002 integrity gate failed: {seed_run['manifest_assertions']}")
    return metrics


def _validate_exp002_config(config: dict[str, Any]) -> None:
    if config["experiment"].get("id") != "EXP002":
        raise ValueError("EXP002 bundle requires experiment.id=EXP002")
    ratios = tuple(float(value) for value in config.get("corruption", {}).get("partial_ratios", ()))
    if ratios != (0.25, 0.5, 0.75):
        raise ValueError("EXP002 requires preregistered partial ratios [0.25, 0.5, 0.75]")
    operations = tuple(config.get("corruption", {}).get("operations", ()))
    if operations != ("relation_flip", "entity_swap", "random_graph"):
        raise ValueError("EXP002 corruption operations differ from preregistration")
    anti = config.get("anti_shortcut", {})
    required = (
        "format_randomization",
        "relation_label_randomization",
        "graph_length_matching",
    )
    if not all(bool(anti.get(key)) for key in required):
        raise ValueError("EXP002 requires all anti-shortcut controls")


def _length_matched_pair(
    model: BaseVLM,
    record: SceneRecord,
    full_graph: SceneGraph,
    random_graph: SceneGraph,
) -> tuple[PreparedInput, PreparedInput]:
    full = _prepare_graph_input(model, record, FULL, full_graph, semantic_hash=_fact_hash(record))
    randomized = _prepare_graph_input(model, record, RANDOM_GRAPH, random_graph)
    full_padding = random_padding = ""
    for _ in range(256):
        if full.input_tokens == randomized.input_tokens:
            return full, randomized
        if full.input_tokens < randomized.input_tokens:
            full_padding = _extend_padding(
                model, record, FULL, full_graph, full_padding, randomized.input_tokens
            )
            full = _prepare_graph_input(
                model,
                record,
                FULL,
                full_graph,
                padding=full_padding,
                semantic_hash=_fact_hash(record),
            )
        else:
            random_padding = _extend_padding(
                model,
                record,
                RANDOM_GRAPH,
                random_graph,
                random_padding,
                full.input_tokens,
            )
            randomized = _prepare_graph_input(
                model,
                record,
                RANDOM_GRAPH,
                random_graph,
                padding=random_padding,
            )
    raise ValueError(f"{record.scene_id}: full/random prompts could not be exactly token matched")


def _extend_padding(
    model: BaseVLM,
    record: SceneRecord,
    condition: str,
    graph: SceneGraph,
    current: str,
    target: int,
) -> str:
    candidates: list[tuple[int, str]] = []
    baseline = _prepare_graph_input(model, record, condition, graph, padding=current)
    for token in PADDING_CANDIDATES:
        candidate = current + token
        prepared = _prepare_graph_input(model, record, condition, graph, padding=candidate)
        increase = prepared.input_tokens - baseline.input_tokens
        if increase > 0 and prepared.input_tokens <= target:
            candidates.append((increase, candidate))
    if not candidates:
        raise ValueError(
            f"{record.scene_id}: tokenizer has no content-free padding step toward {target} tokens"
        )
    return max(candidates, key=lambda item: item[0])[1]


def _prepare_graph_input(
    model: BaseVLM,
    record: SceneRecord,
    condition: str,
    graph: SceneGraph,
    *,
    serialization: str = "canonical",
    relation_labels: dict[str, str] | None = None,
    padding: str = "",
    semantic_hash: str | None = None,
) -> PreparedInput:
    evidence = (
        f"Evidence: scene graph ({serialization}).\n"
        f"{_serialize_graph(graph, relation_labels=relation_labels)}{padding}"
    )
    question = (
        f"Question: {record.question}\n"
        f"Answer with exactly one option: {', '.join(record.choices)}."
    )
    prompt = f"{evidence}\n{question}"
    return PreparedInput(
        prompt=prompt,
        image=record.image or None,
        condition=condition,
        setting="natural",
        input_tokens=model.count_tokens(prompt),
        evidence_tokens=model.count_tokens(evidence),
        semantic_units=len(graph.nodes) * 6 + len(graph.edges),
        relation_count=len(graph.edges),
        semantic_facts_sha256=semantic_hash,
        serialization=serialization,
        padding_units=len(padding.split()),
        token_match_delta=0,
    )


def _serialize_graph(
    graph: SceneGraph, *, relation_labels: dict[str, str] | None = None
) -> str:
    nodes = "\n".join(
        f"- {node['id']}: {node['size']} {node['texture']} {node['color']} "
        f"{node['shape']} [{node['category']}]"
        for node in graph.nodes
    )
    edges = "\n".join(
        f"- {edge['subject']} --{(relation_labels or {}).get(edge['relation'], edge['relation'])}"
        f"--> {edge['object']}"
        for raw in graph.edges
        for edge in (normalize_relation(dict(raw)),)
    )
    return f"Nodes:\n{nodes}\nEdges (Relations):\n{edges}"


def _predict_row(
    model: BaseVLM,
    record: SceneRecord,
    condition: str,
    prepared: PreparedInput,
    *,
    operation: str | None = None,
    ratio: float | None = None,
) -> dict[str, Any]:
    prediction = model.reason(record, condition, prepared_input=prepared)
    input_payload = prepared.to_dict()
    input_payload.update(
        {
            "prompt": prepared.prompt,
            "prompt_sha256": hashlib.sha256(prepared.prompt.encode("utf-8")).hexdigest(),
            "image": str(prepared.image) if prepared.image is not None else None,
        }
    )
    relation = str(record.metadata.get("primary_relation", "unknown"))
    return {
        "scene_id": record.scene_id,
        "condition": condition,
        "answer": record.answer,
        "prediction": prediction,
        "correct": bool(model.evaluate(prediction, record)),
        "seed": record.metadata.get("seed"),
        "split": record.metadata.get("split"),
        "question_type": record.metadata.get("question_type"),
        "hop_depth": record.metadata.get("hop_depth"),
        "primary_relation": relation,
        "relation_group": RELATION_GROUPS.get(relation, "other"),
        "corruption_operation": operation,
        "corruption_ratio": ratio,
        "input": input_payload,
    }


def _opaque_label_map(seed: int) -> dict[str, str]:
    labels = [f"relation_{index:02d}" for index in range(1, len(RELATIONS) + 1)]
    random.Random(seed).shuffle(labels)
    return dict(zip(RELATIONS, labels, strict=True))


def _tag_manifest(
    payload: dict[str, Any], record: SceneRecord, experiment_seed: int
) -> dict[str, Any]:
    return {
        "scene_id": record.scene_id,
        "experiment_seed": experiment_seed,
        "sample_seed": record.metadata.get("sample_seed"),
        "hop_depth": record.metadata.get("hop_depth"),
        "primary_relation": record.metadata.get("primary_relation"),
        **payload,
    }


def _auxiliary_manifest(
    graph: SceneGraph,
    corrupted_graph: dict[str, Any],
    *,
    operation: str,
    seed: int,
    details: dict[str, Any],
) -> dict[str, Any]:
    original = graph.to_dict()
    return {
        "original_graph": original,
        "corrupted_graph": corrupted_graph,
        "operation": operation,
        "seed": seed,
        "ratio": 0.0 if operation == "serialization_order_randomization" else 1.0,
        "selected_edge_indices": list(range(len(graph.edges))),
        "original_graph_sha256": _payload_sha256(original),
        "corrupted_graph_sha256": _payload_sha256(corrupted_graph),
        "preserves_nodes": original["objects"] == corrupted_graph["objects"],
        "preserves_edge_count": len(original["relations"])
        == len(corrupted_graph["relations"]),
        "changed": original != corrupted_graph,
        "details": details,
    }


def _no_control_metadata_in_prompts(rows: list[dict[str, Any]]) -> bool:
    forbidden = ('"original_graph"', '"corrupted_graph"', '"operation"', "corruption_ratio")
    return all(
        not any(value in str(row["input"]["prompt"]) for value in forbidden) for row in rows
    )


def _fact_hash(record: SceneRecord) -> str | None:
    value = record.metadata.get("information_control", {}).get("graph_facts_sha256")
    return str(value) if value else None


def _registered_aggregate(
    registration: dict[str, Any],
    per_seed: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    evaluation = config["evaluation"]
    for index, criterion in enumerate(registration["decision_rule"]["criteria"]):
        path = str(criterion["metric_path"])
        values = [float(_lookup(row["metrics"], path)) for row in per_seed]
        result[path] = multiple_seed_summary(
            values,
            confidence=float(evaluation["confidence_level"]),
            bootstrap_samples=int(evaluation["bootstrap_samples"]),
            seed=int(evaluation["statistics_seed"]) + index,
            significance_test="paired_bootstrap",
        )
    return result


def _descriptive_aggregate(
    per_seed: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    ratios = tuple(float(value) for value in config["corruption"]["partial_ratios"])
    samples = int(config["evaluation"]["bootstrap_samples"])
    seed = int(config["evaluation"]["statistics_seed"])
    condition_names = (
        IMAGE_ONLY,
        FULL,
        *(partial_condition(value) for value in ratios),
        RELATION_FLIP,
        OBJECT_SWAP,
        RANDOM_GRAPH,
        ORDER_RANDOMIZED,
        LABEL_RANDOMIZED,
    )
    conditions = {
        condition: _summary(
            per_seed,
            f"conditions.{condition}.accuracy",
            samples=samples,
            seed=seed + index,
        )
        for index, condition in enumerate(condition_names)
    }
    partial = {
        str(int(round(ratio * 100))): {
            "full": conditions[FULL],
            "partial": conditions[partial_condition(ratio)],
            "structural_dependency": _summary(
                per_seed,
                f"structural_dependency.full_over_partial_{int(round(ratio * 100)):02d}.estimate",
                samples=samples,
                seed=seed + 100 + index,
            ),
        }
        for index, ratio in enumerate(ratios)
    }
    depths: dict[str, Any] = {}
    for depth in range(1, 5):
        depths[str(depth)] = {
            condition: _summary(
                per_seed,
                f"reasoning_depth.{depth}.conditions.{condition}.accuracy",
                samples=samples,
                seed=seed + depth * 20 + index,
            )
            for index, condition in enumerate((FULL, partial_condition(0.5), RANDOM_GRAPH))
        }
        depths[str(depth)]["full_over_random_graph"] = _summary(
            per_seed,
            f"reasoning_depth.{depth}.structural_dependency.full_over_random_graph.estimate",
            samples=samples,
            seed=seed + depth * 20 + 10,
        )
    relations: dict[str, Any] = {}
    for index, relation in enumerate(config["evaluation"]["relation_breakdown"]):
        relations[str(relation)] = {
            "full": _summary(
                per_seed,
                f"relation_breakdown.relations.{relation}.conditions.{FULL}.accuracy",
                samples=samples,
                seed=seed + 300 + index,
            ),
            "random": _summary(
                per_seed,
                f"relation_breakdown.relations.{relation}.conditions.{RANDOM_GRAPH}.accuracy",
                samples=samples,
                seed=seed + 320 + index,
            ),
            "structural_dependency": _summary(
                per_seed,
                f"relation_breakdown.relations.{relation}.full_over_random_graph.estimate",
                samples=samples,
                seed=seed + 340 + index,
            ),
        }
    return {
        "condition_summaries": conditions,
        "partial_ratio_summaries": partial,
        "reasoning_depth_summaries": depths,
        "relation_summaries": relations,
        "anti_shortcut_summaries": {
            "format_full_over_randomized": _summary(
                per_seed,
                "anti_shortcut.format_randomization.estimate",
                samples=samples,
                seed=seed + 400,
            ),
            "full_over_opaque_labels": _summary(
                per_seed,
                "anti_shortcut.relation_label_randomization.estimate",
                samples=samples,
                seed=seed + 401,
            ),
        },
    }


def _summary(
    per_seed: list[dict[str, Any]], path: str, *, samples: int, seed: int
) -> dict[str, Any]:
    values = [float(_lookup(row["metrics"], path)) for row in per_seed]
    return multiple_seed_summary(
        values,
        bootstrap_samples=samples,
        seed=seed,
        significance_test="paired_bootstrap",
    )


def _seed_integrity(seed_dir: Path, seed: int) -> dict[str, Any]:
    dataset_path = seed_dir / "dataset" / "manifest.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    run = json.loads((seed_dir / "run.json").read_text(encoding="utf-8"))
    return {
        "seed": seed,
        "dataset_manifest": dataset_path.as_posix(),
        "dataset_manifest_sha256": _sha256(dataset_path),
        "corruption_manifest_sha256": _sha256(seed_dir / "corruption_manifest.json"),
        "manifest_assertions": {
            **run["manifest_assertions"],
            "generator": dataset.get("generator"),
        },
    }


def _scientific_interpretation(report: dict[str, Any], config: dict[str, Any]) -> str:
    if config["model"]["backend"] == "reference":
        return (
            "INCONCLUSIVE: ReferenceVLM validates corruption, pairing, statistics, and artifacts; "
            "it is not eligible evidence that a real VLM depends on structural relations."
        )
    if report["final_decision"] == "GO":
        return "Every preregistered necessity, corruption, depth, and seed-stability gate passed."
    if report["final_decision"] == "NO-GO":
        return "At least one structural-necessity falsification gate failed."
    return "INCONCLUSIVE: required model evidence or an integrity gate is incomplete."


def _write_figures(
    output: Path, descriptive: dict[str, Any], config: dict[str, Any]
) -> None:
    conditions = descriptive["condition_summaries"]
    ratios = tuple(float(value) for value in config["corruption"]["partial_ratios"])
    degradation_values = [
        float(conditions[FULL]["mean"]),
        *(float(conditions[partial_condition(ratio)]["mean"]) for ratio in ratios),
        float(conditions[RANDOM_GRAPH]["mean"]),
    ]
    _line_chart(
        output / "figure1_accuracy_degradation.png",
        "EXP002 Accuracy Degradation",
        ["Full", "Partial 25%", "Partial 50%", "Partial 75%", "Random"],
        {"Accuracy": degradation_values},
    )
    _line_chart(
        output / "figure2_corruption_ratio.png",
        "Performance vs Relation Removal Ratio",
        ["0%", "25%", "50%", "75%"],
        {
            "Accuracy": [
                float(conditions[FULL]["mean"]),
                *(float(conditions[partial_condition(ratio)]["mean"]) for ratio in ratios),
            ]
        },
    )
    depths = descriptive["reasoning_depth_summaries"]
    _line_chart(
        output / "figure3_reasoning_depth_degradation.png",
        "Reasoning Depth Degradation",
        ["1-hop", "2-hop", "3-hop", "4-hop"],
        {
            "Full": [float(depths[str(value)][FULL]["mean"]) for value in range(1, 5)],
            "Partial 50%": [
                float(depths[str(value)][partial_condition(0.5)]["mean"])
                for value in range(1, 5)
            ],
            "Random": [
                float(depths[str(value)][RANDOM_GRAPH]["mean"]) for value in range(1, 5)
            ],
        },
    )


def _line_chart(
    path: Path, title: str, labels: list[str], series: dict[str, list[float]]
) -> None:
    image = Image.new("RGB", (1000, 620), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 90, 70, 950, 520
    draw.line((left, top, left, bottom, right, bottom), fill="black", width=2)
    draw.text((left, 25), title, fill="black")
    for tick in range(6):
        value = tick / 5
        y = bottom - value * (bottom - top)
        draw.line((left - 5, y, right, y), fill="#e5e7eb")
        draw.text((45, y - 7), f"{value:.1f}", fill="black")
    colors = ("#2563eb", "#f59e0b", "#dc2626", "#059669")
    width = (right - left) / max(1, len(labels) - 1)
    for series_index, (name, values) in enumerate(series.items()):
        points = [
            (left + index * width, bottom - float(value) * (bottom - top))
            for index, value in enumerate(values)
        ]
        color = colors[series_index % len(colors)]
        if len(points) > 1:
            draw.line(points, fill=color, width=4)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        draw.text((right - 170, top + series_index * 22), name, fill=color)
    for index, label in enumerate(labels):
        draw.text((left + index * width - 25, bottom + 15), label, fill="black")
    image.save(path)


def _write_relation_table(output: Path, descriptive: dict[str, Any]) -> None:
    with (output / "relation_breakdown.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["relation", "full_accuracy", "random_accuracy", "SDS_full_minus_random"])
        for relation, values in descriptive["relation_summaries"].items():
            writer.writerow(
                [
                    relation,
                    values["full"]["mean"],
                    values["random"]["mean"],
                    values["structural_dependency"]["mean"],
                ]
            )


def _write_narrative_reports(
    output: Path,
    result: dict[str, Any],
    report: dict[str, Any],
    config: dict[str, Any],
) -> None:
    (output / "implementation_report.md").write_text(
        "# EXP002 Implementation Report\n\n"
        "Implemented deterministic relation removal at 25/50/75%, relation flips, entity "
        "swaps, same-size random graphs, serialization-order randomization, opaque relation "
        "labels, exact tokenizer-level Full/Random length matching, paired bootstrap inference, "
        "depth/relation slices, integrity manifests, figures, and governed decisions.\n",
        encoding="utf-8",
        newline="\n",
    )
    aggregate = result["metrics"]["aggregate"]
    lines = [
        "# Structural Necessity Analysis",
        "",
        f"- Final governed decision: **{report['final_decision']}**",
        f"- Backend: `{config['model']['backend']}`",
        "- Structural Dependency Score (SDS) is Full accuracy minus the named control.",
        "",
        "| Registered effect | Mean SDS | 95% CI | p-value |",
        "|---|---:|---:|---:|",
    ]
    for path, summary in aggregate.items():
        interval = summary["confidence_interval"]
        lines.append(
            f"| `{path}` | {summary['mean']:.4f} | "
            f"[{interval['lower']:.4f}, {interval['upper']:.4f}] | "
            f"{summary['statistical_test']['p_value']:.4g} |"
        )
    lines.extend(
        [
            "",
            "Reference-backend results are infrastructure validation only and cannot establish "
            "the mechanism claim.",
            "",
        ]
    )
    (output / "structural_necessity_analysis.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )
    (output / "corruption_evaluation_report.md").write_text(
        "# Corruption Evaluation Report\n\n"
        "Every mutation is recorded in `corruption_manifest.json` with original/corrupted "
        "graphs, hashes, seed, ratio, changed edge indices, and invariant checks. Corrupted "
        "graphs preserve the node inventory; flip, swap, random, order, and label controls "
        "preserve edge count. Full and Random prompts have an exact zero-token delta.\n",
        encoding="utf-8",
        newline="\n",
    )


def _record_failed_bundle(
    output: Path,
    config: dict[str, Any],
    seeds: list[int],
    per_seed: list[dict[str, Any]],
    integrity_rows: list[dict[str, Any]],
    error: Exception,
) -> None:
    result = {
        "schema_version": 1,
        "run_id": f"exp002-failed-{utc_now().replace(':', '').replace('-', '')}",
        "experiment_id": "EXP002",
        "hypothesis": "H002",
        "model": f"frozen-{config['model']['backend']}",
        "dataset": f"{config['data']['dataset']}@{config['data']['version']}",
        "seed": seeds,
        "metrics": {"per_seed": per_seed, "aggregate": {}},
        "confidence_interval": None,
        "statistical_test": None,
        "decision": "INCONCLUSIVE",
        "timestamp": utc_now(),
        "integrity": {"registry_validated": True, "per_seed": integrity_rows},
        "status": "failed",
    }
    _write_json(output / "metrics.json", result)
    _write_json(
        output / "run.json",
        {
            "experiment_id": "EXP002",
            "status": "failed",
            "decision": "INCONCLUSIVE",
            "error": f"{type(error).__name__}: {error}",
            "completed_seeds": [row["seed"] for row in per_seed],
        },
    )


def _bundle_manifest(output: Path, run_record: dict[str, Any]) -> dict[str, Any]:
    artifacts = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return {
        "schema_version": 1,
        "experiment_id": "EXP002",
        "run_id": run_record["run_id"],
        "status": run_record["status"],
        "artifacts": artifacts,
    }


def _ensure_new_bundle(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"EXP002 output bundle already exists and is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)


def _lookup(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"missing registered metric path: {path}")
        current = current[part]
    return current


def _payload_sha256(payload: dict[str, Any]) -> str:
    value = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


__all__ = ["run", "run_seed_config"]
