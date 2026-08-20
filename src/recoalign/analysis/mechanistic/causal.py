"""Causal token interventions that preserve tensor shape and provenance."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from recoalign.models.recoalign import ReCoAlignModel

INTERVENTIONS = ("remove", "shuffle", "replace")


def intervene_structure_tokens(
    structure_tokens: Tensor,
    intervention: str,
    *,
    replacement: Tensor | None = None,
    seed: int = 0,
) -> Tensor:
    """Return an intervention tensor; never mutates the supplied representation."""

    if structure_tokens.ndim != 3:
        raise ValueError("structure_tokens must have shape [batch, slots, dimension]")
    if intervention not in INTERVENTIONS:
        raise ValueError(f"unknown token intervention: {intervention}")
    if intervention == "remove":
        return torch.zeros_like(structure_tokens)
    if intervention == "replace":
        if replacement is None or replacement.shape != structure_tokens.shape:
            raise ValueError("replace intervention requires a same-shaped replacement tensor")
        return replacement.detach().clone().to(structure_tokens)
    generator = torch.Generator(device=structure_tokens.device).manual_seed(seed)
    permutations = torch.stack(
        [
            torch.randperm(
                structure_tokens.shape[1],
                generator=generator,
                device=structure_tokens.device,
            )
            for _ in range(structure_tokens.shape[0])
        ]
    )
    return torch.stack(
        [
            structure_tokens[index].index_select(0, permutations[index])
            for index in range(structure_tokens.shape[0])
        ]
    )


def evaluate_token_interventions(
    model: ReCoAlignModel,
    visual_tokens: Tensor,
    answer_labels: Tensor,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Measure baseline and intervention accuracy on paired inputs."""

    model.eval()
    with torch.no_grad():
        baseline = model(visual_tokens)
        original_tokens = baseline["structure_tokens"]
        baseline_accuracy = _accuracy(baseline["reasoning_logits"], answer_labels)
        rows: dict[str, Any] = {
            "baseline": {"accuracy": baseline_accuracy, "changed_fraction": 0.0},
            "interventions": {},
            "seed": seed,
            "oracle_graph_used": False,
        }
        for intervention in INTERVENTIONS:
            replacement = original_tokens.roll(1, dims=0) if intervention == "replace" else None
            modified = intervene_structure_tokens(
                original_tokens,
                intervention,
                replacement=replacement,
                seed=seed,
            )
            output = model.forward_with_structure_tokens(visual_tokens, modified)
            predictions = output["reasoning_logits"].argmax(dim=-1)
            baseline_predictions = baseline["reasoning_logits"].argmax(dim=-1)
            accuracy = _accuracy(output["reasoning_logits"], answer_labels)
            rows["interventions"][intervention] = {
                "accuracy": accuracy,
                "accuracy_delta": accuracy - baseline_accuracy,
                "changed_fraction": float((predictions != baseline_predictions).float().mean()),
                "token_l2_delta": float((modified - original_tokens).pow(2).mean().sqrt()),
            }
    return rows


def _accuracy(logits: Tensor, labels: Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels.to(logits.device)).float().mean().cpu())


__all__ = ["INTERVENTIONS", "evaluate_token_interventions", "intervene_structure_tokens"]
