"""Execute the preregistered three-stage structured-interface diagnosis."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from pathlib import Path
from typing import Any

import yaml

from diagnosis.interface_diagnosis.probes import consistency_score, graph_score, linear_probe
from evaluation.statistics import multiple_seed_summary
from experiments.runtime import build_generator, load_config, run_conditions
from recoalign.models.vlm.registry import get_model_registry
from recoalign.reproducibility import get_git_metadata, utc_now
from recoalign.research_registry import get_registered_experiment, validate_research_registries
from recoalign.schema_validation import repository_root, validate_payload

REPRESENTATIONS = ("image_encoder", "projected_visual_token", "llm_visual_hidden")
LABELS = ("object", "attribute", "relation", "composition")


def run_interface_diagnosis(
    *,
    model_name: str = "reference",
    experiment_id: str = "EXP004",
    config_path: str | Path = "configs/diagnosis/interface_diagnosis.yaml",
    output_dir: str | Path | None = None,
    seeds: list[int] | None = None,
    dry_run: bool = False,
) -> Path:
    """Run EXP004 or materialize its validated dry-run bundle."""

    project = repository_root()
    validate_research_registries(project)
    registration = get_registered_experiment(experiment_id, project)
    config = load_config(config_path)
    if config["experiment"].get("id") != experiment_id:
        raise ValueError("diagnosis config does not match the registered experiment")
    definition = get_model_registry().definition(model_name)
    resolved = dict(config)
    resolved["model"] = definition.experiment_model_config()
    resolved["model"]["seed"] = int(config["experiment"]["seed"])
    selected_seeds = list(seeds or config["experiment"]["seeds"])
    _validate_seeds(selected_seeds)
    destination = Path(output_dir or config["experiment"]["output_dir"])
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"diagnosis output is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    resolved["experiment"] = dict(resolved["experiment"])
    resolved["experiment"]["output_dir"] = destination.as_posix()
    (destination / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8", newline="\n"
    )
    if dry_run:
        _write_dry_run(destination, registration, definition, selected_seeds)
        return destination

    per_seed: list[dict[str, Any]] = []
    for seed in selected_seeds:
        per_seed.append(
            _run_seed(
                resolved,
                model_name=model_name,
                seed=seed,
                output=destination / "seeds" / str(seed),
            )
        )
    aggregate = _aggregate(per_seed, resolved)
    result = {
        "schema_version": 1,
        "run_id": f"EXP004-{uuid.uuid4().hex[:12]}",
        "experiment_id": "EXP004",
        "hypothesis": "H004",
        "model": f"frozen-{model_name}",
        "dataset": f"{registration['dataset']['name']}@{registration['dataset']['version']}",
        "seed": selected_seeds,
        "status": "complete",
        "metrics": {"per_seed": per_seed, "aggregate": aggregate},
        "diagnosis": _diagnosis_summary(aggregate, resolved, model_name),
        "integrity": {
            "registry_validated": True,
            "per_seed": [_seed_integrity(row) for row in per_seed],
        },
        "timestamp": utc_now(),
    }
    report = _decision_report(registration, result)
    result["decision"] = report["final_decision"]
    validate_payload("interface_diagnosis", result)
    (destination / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (destination / "analysis.json").write_text(
        json.dumps(result["diagnosis"], indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (destination / "decision_report.yaml").write_text(
        yaml.safe_dump(report, sort_keys=False), encoding="utf-8", newline="\n"
    )
    (destination / "run.json").write_text(
        json.dumps(
            {
                "experiment_id": "EXP004",
                "run_id": result["run_id"],
                "model": result["model"],
                "status": result["status"],
                "decision": result["decision"],
                "git": get_git_metadata(project),
                "model_provenance": _safe_model_provenance(model_name),
                "config_sha256": _sha256(destination / "config.resolved.yaml"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_figures(destination, result["diagnosis"])
    return destination


def _run_seed(
    config: dict[str, Any], *, model_name: str, seed: int, output: Path
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    resolved = json.loads(json.dumps(config))
    resolved["experiment"]["seed"] = seed
    resolved["model"]["seed"] = seed
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8", newline="\n"
    )
    generator = build_generator(resolved)
    records = generator.generate(
        int(resolved["synthetic"].get("count", 24)),
        output_dir=output / "dataset",
        seed=seed,
        split="diagnosis",
    )
    model = get_model_registry().get_or_create(resolved["model"], seed=seed)
    stage1 = _stage_one(model, records, seed)
    stage2 = _stage_two(model, records, stage1, seed)
    stage3 = _stage_three(model, records, resolved, seed)
    scores = _scores(stage1, stage2, stage3)
    payload = {
        "seed": seed,
        "dataset_size": len(records),
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
        "scores": scores,
        "manifest_assertions": {
            "semantic_probe_scene_disjoint": True,
            "training_enabled": False,
            "oracle_graph_is_external": True,
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload


def _stage_one(model: Any, records: list[Any], seed: int) -> dict[str, Any]:
    images = [record.image for record in records]
    labels = _labels(records)
    result: dict[str, Any] = {"stage": "visual_semantic_availability", "representations": {}}
    for representation in REPRESENTATIONS:
        try:
            if representation == "image_encoder":
                features = model.encode_image(images)
            elif representation == "projected_visual_token":
                features = model.encode_visual_tokens(images)
            else:
                features = model.encode_visual_hidden(images)
            feature_values = _flatten_features(features)
            result["representations"][representation] = {
                "status": "available",
                "probes": {
                    label: linear_probe(feature_values, values, seed=seed + index)
                    for index, (label, values) in enumerate(labels.items())
                },
            }
        except (AttributeError, NotImplementedError, RuntimeError, ValueError, OSError) as exc:
            result["representations"][representation] = {
                "status": "unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
            }
    available = [
        row for row in result["representations"].values() if row.get("status") == "available"
    ]
    probe_scores = [
        float(probe["accuracy"])
        for row in available
        for probe in row["probes"].values()
        if probe.get("status") == "available"
    ]
    result["SAS"] = float(sum(probe_scores) / len(probe_scores)) if probe_scores else None
    result["available_representation_count"] = len(available)
    return result


def _stage_two(model: Any, records: list[Any], stage1: dict[str, Any], seed: int) -> dict[str, Any]:
    images = [record.image for record in records]
    labels = _labels(records)
    result: dict[str, Any] = {
        "stage": "structured_representation_accessibility",
        "graph_reconstruction": {"status": "unavailable", "reason": "adapter has no method"},
    }
    graph_rows: list[dict[str, Any]] = []
    for record in records:
        try:
            predicted = model.reconstruct_graph(record)
            graph_rows.append(
                graph_score(predicted, {"objects": record.objects, "relations": record.relations})
            )
        except (AttributeError, NotImplementedError, RuntimeError, ValueError, OSError) as exc:
            result["graph_reconstruction"] = {
                "status": "unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            graph_rows = []
            break
    if graph_rows:
        result["graph_reconstruction"] = {
            "status": "available",
            "node_f1": _mean(row["node_f1"] for row in graph_rows),
            "relation_f1": _mean(row["relation_f1"] for row in graph_rows),
            "graph_edit_distance": _mean(row["graph_edit_distance"] for row in graph_rows),
            "exact_rate": _mean(float(row["exact"]) for row in graph_rows),
        }
    feature_row = stage1["representations"].get("projected_visual_token")
    if feature_row is None or feature_row.get("status") != "available":
        feature_row = stage1["representations"].get("image_encoder")
    relation_probe = None
    consistency = None
    try:
        if feature_row and feature_row.get("status") == "available":
            if feature_row is stage1["representations"].get("image_encoder"):
                features = model.encode_image(images)
            else:
                features = model.encode_visual_tokens(images)
            values = _flatten_features(features)
            relation_probe = linear_probe(values, labels["relation"], seed=seed + 101)
            consistency = consistency_score(values, labels["relation"])
    except (AttributeError, NotImplementedError, RuntimeError, ValueError, OSError) as exc:
        result["representation_probe_error"] = f"{type(exc).__name__}: {exc}"
    result["latent_relation_probe"] = relation_probe or {"status": "unavailable"}
    result["structure_consistency"] = consistency or {"status": "unavailable"}
    components = []
    if result["graph_reconstruction"].get("status") == "available":
        components.append(float(result["graph_reconstruction"]["relation_f1"]))
    if relation_probe and relation_probe.get("status") == "available":
        components.append(float(relation_probe["accuracy"]))
    if consistency and consistency.get("status") == "available":
        components.append((float(consistency["consistency_gap"]) + 1.0) / 2.0)
    result["StAS"] = _mean(components) if components else None
    result["available_measurement_count"] = len(components)
    return result


def _stage_three(
    model: Any, records: list[Any], config: dict[str, Any], seed: int
) -> dict[str, Any]:
    conditions = tuple(config["evaluation"]["conditions"])
    rows = run_conditions(records, model, conditions)
    accuracies = {
        condition: _mean(float(row["correct"]) for row in rows if row["condition"] == condition)
        for condition in conditions
    }
    result = {
        "stage": "reasoning_execution",
        "conditions": accuracies,
        "prediction_count": len(rows),
        "oracle_graph_accuracy": accuracies.get("scene_graph"),
        "image_only_accuracy": accuracies.get("image_only"),
        "corrupted_graph_accuracy": accuracies.get("corrupted_graph"),
        "oracle_graph_gain": _difference(
            accuracies.get("scene_graph"), accuracies.get("image_only")
        ),
        "noise_sensitivity": _difference(
            accuracies.get("scene_graph"), accuracies.get("corrupted_graph")
        ),
        "failure_taxonomy": _failure_taxonomy(rows),
    }
    result["RES"] = result["oracle_graph_accuracy"]
    (
        Path(config["experiment"]["output_dir"]) / "seeds" / str(seed) / "predictions.jsonl"
    ).write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def _scores(
    stage1: dict[str, Any], stage2: dict[str, Any], stage3: dict[str, Any]
) -> dict[str, float | None]:
    sas = stage1.get("SAS")
    stas = stage2.get("StAS")
    res = stage3.get("RES")
    return {
        "SAS": sas,
        "StAS": stas,
        "RES": res,
        "interface_gap": _difference(sas, stas),
        "oracle_graph_gain": stage3.get("oracle_graph_gain"),
    }


def _aggregate(per_seed: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for metric in ("SAS", "StAS", "RES", "interface_gap", "oracle_graph_gain"):
        values = [row["scores"][metric] for row in per_seed]
        numeric = [
            float(value) for value in values if value is not None and math.isfinite(float(value))
        ]
        if numeric:
            aggregate[f"diagnosis.{metric.lower()}"] = multiple_seed_summary(
                numeric,
                confidence=float(config["evaluation"]["confidence_level"]),
                bootstrap_samples=int(config["evaluation"]["bootstrap_samples"]),
                seed=int(config["evaluation"]["statistics_seed"]) + len(metric),
            )
    taxonomy: dict[str, int] = {}
    for row in per_seed:
        for category, count in row["stage3"].get("failure_taxonomy", {}).items():
            taxonomy[category] = taxonomy.get(category, 0) + int(count)
    aggregate["failure_taxonomy"] = taxonomy
    return aggregate


def _diagnosis_summary(
    aggregate: dict[str, Any], config: dict[str, Any], model_name: str
) -> dict[str, Any]:
    values = {key.removeprefix("diagnosis."): value for key, value in aggregate.items()}
    sas = values.get("sas", {}).get("mean")
    stas = values.get("stas", {}).get("mean")
    res = values.get("res", {}).get("mean")
    gap = values.get("interface_gap", {}).get("mean")
    threshold = config.get("diagnosis", {}).get("thresholds", {})
    if sas is None or stas is None or res is None:
        classification = "INCONCLUSIVE_UNAVAILABLE_MEASUREMENT"
    elif sas < float(threshold.get("semantic_low", 0.5)):
        classification = "TYPE_A_SEMANTIC_FAILURE"
    elif stas < float(threshold.get("structured_low", 0.6)) or float(gap or 0.0) >= float(
        threshold.get("interface_gap", 0.1)
    ):
        classification = "TYPE_B_STRUCTURE_INTERFACE_FAILURE"
    elif res < float(threshold.get("reasoning_low", 0.5)):
        classification = "TYPE_C_REASONING_EXECUTION_FAILURE"
    else:
        classification = "NO_CLEAR_MECHANISTIC_GAP"
    return {
        "model": model_name,
        "scores": values,
        "failure_taxonomy": aggregate.get("failure_taxonomy", {}),
        "classification": classification,
        "interpretation": _interpretation(classification),
        "claim_status": "infrastructure_validation"
        if model_name == "reference"
        else "requires_review",
    }


def _decision_report(registration: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    from research.decision_engine import evaluate_decision

    report = evaluate_decision(registration, result)
    validate_payload("decision_report", report)
    return report


def _stage_integrity_assertions() -> dict[str, Any]:
    return {
        "semantic_probe_scene_disjoint": True,
        "training_enabled": False,
        "oracle_graph_is_external": True,
    }


def _seed_integrity(row: dict[str, Any]) -> dict[str, Any]:
    return {"seed": row["seed"], "manifest_assertions": row["manifest_assertions"]}


def _labels(records: list[Any]) -> dict[str, list[str]]:
    return {
        "object": [
            ";".join(
                sorted(
                    str(node.get("category", node.get("shape", "object")))
                    for node in record.objects
                )
            )
            for record in records
        ],
        "attribute": [
            ";".join(sorted(str(node.get("color", "")) for node in record.objects))
            for record in records
        ],
        "relation": [
            ";".join(sorted(str(edge.get("relation", "")) for edge in record.relations))
            for record in records
        ],
        "composition": [str(record.metadata.get("composition", "unknown")) for record in records],
    }


def _flatten_features(features: Any) -> Any:
    values = __import__("numpy").asarray(features, dtype=float)
    if values.ndim < 2:
        raise ValueError("representation must have a batch dimension")
    return values.reshape(values.shape[0], -1)


def _mean(values: Any) -> float:
    numbers = [float(value) for value in values]
    return sum(numbers) / len(numbers) if numbers else 0.0


def _difference(first: Any, second: Any) -> float | None:
    return None if first is None or second is None else float(first) - float(second)


def _interpretation(classification: str) -> str:
    return {
        "TYPE_A_SEMANTIC_FAILURE": (
            "Visual semantic probes are low; an interface diagnosis is not supported."
        ),
        "TYPE_B_STRUCTURE_INTERFACE_FAILURE": (
            "Semantic evidence is more accessible than structured relational evidence."
        ),
        "TYPE_C_REASONING_EXECUTION_FAILURE": (
            "Semantic and structural probes are high, but oracle-graph reasoning remains low."
        ),
        "NO_CLEAR_MECHANISTIC_GAP": (
            "The three scores do not support a single registered failure type."
        ),
        "INCONCLUSIVE_UNAVAILABLE_MEASUREMENT": (
            "At least one required stage measurement is unavailable."
        ),
    }.get(classification, "Unclassified diagnostic outcome.")


def _safe_model_provenance(model_name: str) -> dict[str, Any]:
    try:
        definition = get_model_registry().definition(model_name)
        return {
            "model_config": definition.path.as_posix(),
            "model_config_sha256": definition.sha256,
            "adapter": definition.payload["adapter"],
        }
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}


def _write_dry_run(
    destination: Path, registration: dict[str, Any], definition: Any, seeds: list[int]
) -> None:
    payload = {
        "schema_version": 1,
        "experiment_id": "EXP004",
        "status": "dry-run",
        "model": definition.name,
        "seeds": seeds,
        "training_enabled": False,
        "registered_conditions": registration["input_conditions"],
        "required_outputs": ["metrics.json", "analysis.json", "decision_report.yaml"],
        "scientific_decision": "INCONCLUSIVE",
        "diagnosis": {"status": "dry-run", "measurements": REPRESENTATIONS},
    }
    (destination / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    (destination / "analysis.json").write_text(
        json.dumps({"status": "dry-run", "measurements": REPRESENTATIONS}, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (destination / "decision_report.yaml").write_text(
        yaml.safe_dump(
            {"experiment_id": "EXP004", "final_decision": "INCONCLUSIVE"}, sort_keys=False
        ),
        encoding="utf-8",
        newline="\n",
    )


def _write_figures(destination: Path, diagnosis: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    scores = diagnosis.get("scores", {})
    pipeline, pipeline_axis = plt.subplots(figsize=(8, 2.4))
    pipeline_axis.axis("off")
    boxes = [
        (0.08, "Visual\nsemantics"),
        (0.39, "Structured\naccessibility"),
        (0.70, "Reasoning\nexecution"),
    ]
    for index, (x, label) in enumerate(boxes):
        pipeline_axis.text(
            x,
            0.5,
            label,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.8", "facecolor": "#dbeafe"},
            transform=pipeline_axis.transAxes,
        )
        if index < len(boxes) - 1:
            pipeline_axis.annotate(
                "",
                xy=(boxes[index + 1][0] - 0.08, 0.5),
                xytext=(x + 0.08, 0.5),
                xycoords=pipeline_axis.transAxes,
                arrowprops={"arrowstyle": "->", "lw": 2},
            )
    pipeline_axis.set_title("EXP004 three-stage diagnostic pipeline")
    pipeline.tight_layout()
    pipeline.savefig(destination / "figure1_three_stage_pipeline.png", dpi=140)
    plt.close(pipeline)

    figure, axis = plt.subplots(figsize=(6, 4))
    names = ["SAS", "StAS", "RES"]
    axis.bar(names, [float(scores.get(name.lower(), {}).get("mean", 0.0) or 0.0) for name in names])
    axis.set_ylim(0, 1)
    axis.set_ylabel("score")
    axis.set_title("EXP004 three-stage diagnostic matrix")
    figure.tight_layout()
    figure.savefig(destination / "figure2_diagnostic_matrix.png", dpi=140)
    plt.close(figure)

    taxonomy = diagnosis.get("failure_taxonomy", {})
    failure_figure, failure_axis = plt.subplots(figsize=(6, 4))
    names = list(taxonomy) or ["none"]
    failure_axis.bar(names, [taxonomy.get(name, 0) for name in names], color="#f59e0b")
    failure_axis.set_ylabel("error count")
    failure_axis.set_title("EXP004 reasoning failure taxonomy")
    failure_axis.tick_params(axis="x", rotation=25)
    failure_figure.tight_layout()
    failure_figure.savefig(destination / "figure3_failure_taxonomy.png", dpi=140)
    plt.close(failure_figure)


def _failure_taxonomy(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "object_failure": 0,
        "relation_failure": 0,
        "multi_hop_failure": 0,
        "structure_misuse": 0,
    }
    for row in rows:
        if row.get("correct"):
            continue
        if row.get("condition") == "corrupted_graph":
            counts["structure_misuse"] += 1
        elif int(row.get("hop_depth") or 1) > 1:
            counts["multi_hop_failure"] += 1
        elif row.get("question_type") == "relation_reasoning":
            counts["relation_failure"] += 1
        else:
            counts["object_failure"] += 1
    return counts


def _validate_seeds(seeds: list[int]) -> None:
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ValueError("seeds must be unique non-negative integers")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["run_interface_diagnosis"]
