"""Toy mechanistic suite exercising every attribution family."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from recoalign.models.recoalign import ReCoAlignConfig, ReCoAlignModel
from recoalign.training.recoalign_toy import run_toy_training

from .causal import evaluate_token_interventions
from .parameters import parameter_audit, parameter_matched_control
from .probes import representation_comparison
from .registry import ABLATION_SPECS
from .statistics import summarize_ablation_seeds
from .visualization import attention_summary, token_clusters


def run_toy_mechanistic_suite(
    *,
    output_dir: str | Path = "reports/mechanistic",
    seeds: tuple[int, ...] = (101, 202, 303),
) -> dict[str, Any]:
    """Run controls on the same deterministic toy split; never claim benchmark efficacy."""

    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("mechanistic suite requires at least three unique seeds")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    base = ReCoAlignConfig(
        visual_dim=8,
        interface_dim=16,
        llm_dim=12,
        num_structure_tokens=4,
        num_heads=4,
        num_object_classes=2,
        num_attribute_classes=2,
        num_relation_classes=2,
        num_composition_classes=2,
        num_answer_classes=2,
    )
    values: dict[str, list[float]] = {"full": [], "no_structure": [], "random_structure": []}
    intervention_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for seed in seeds:
        full = run_toy_training(base, steps=20, batch_size=8, seed=seed)
        no_structure = run_toy_training(
            ReCoAlignConfig(
                **{
                    **base.to_dict(),
                    "use_structure_tokens": False,
                    "random_structure_tokens": False,
                }
            ),
            steps=20,
            batch_size=8,
            seed=seed,
        )
        random_structure = run_toy_training(
            ReCoAlignConfig(**{**base.to_dict(), "random_structure_tokens": True}),
            steps=20,
            batch_size=8,
            seed=seed,
        )
        values["full"].append(
            _accuracy(full["model"], full["batch"].answer_labels, full["batch"].visual_tokens)
        )
        values["no_structure"].append(
            _accuracy(
                no_structure["model"],
                no_structure["batch"].answer_labels,
                no_structure["batch"].visual_tokens,
            )
        )
        values["random_structure"].append(
            _accuracy(
                random_structure["model"],
                random_structure["batch"].answer_labels,
                random_structure["batch"].visual_tokens,
            )
        )
        model = full["model"].eval()
        batch = full["batch"]
        with torch.no_grad():
            output = model(batch.visual_tokens)
        intervention_rows.append(
            evaluate_token_interventions(model, batch.visual_tokens, batch.answer_labels, seed=seed)
        )
        with torch.no_grad():
            structure = output["structure_tokens"].cpu().numpy()
            visual = model.interface.encoder.project_visual(batch.visual_tokens).cpu().numpy()
            attention = output["reasoning_attention"].cpu().numpy()
        labels = {
            "object": batch.object_targets.cpu().numpy(),
            "attribute": batch.attribute_targets.cpu().numpy(),
            "relation": batch.relation_targets.cpu().numpy(),
            "composition": batch.composition_targets.cpu().numpy(),
        }
        probe_rows.append(
            {
                "seed": seed,
                "representation": representation_comparison(
                    visual, structure, labels, seed=seed
                ),
                "clusters": token_clusters(structure),
                "attention": attention_summary(
                    attention, structure_token_start=batch.visual_tokens.shape[1]
                ),
            }
        )
        _control, audit = parameter_matched_control(model, input_dim=8, output_dim=16)
        parameter_rows.append(
            {"seed": seed, "recoalign": parameter_audit(model), "matched_control": audit}
        )
    result = {
        "schema_version": 1,
        "seeds": list(seeds),
        "ablation_statistics": {
            name: summarize_ablation_seeds(values["full"], values[name], seed_ids=seeds)
            for name in ("no_structure", "random_structure")
        },
        "causal_interventions": intervention_rows,
        "representation": probe_rows,
        "parameter_controls": parameter_rows,
        "registered_ablation_count": len(ABLATION_SPECS),
        "oracle_graph_used": False,
        "scientific_status": "toy_mechanistic_validation_only",
    }
    (destination / "mechanistic_suite.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def _accuracy(model: ReCoAlignModel, labels: torch.Tensor, visual_tokens: torch.Tensor) -> float:
    with torch.no_grad():
        logits = model(visual_tokens)["reasoning_logits"]
    return float((logits.argmax(-1) == labels).float().mean())


__all__ = ["run_toy_mechanistic_suite"]
