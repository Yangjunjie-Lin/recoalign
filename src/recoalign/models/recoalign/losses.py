"""The three theory-motivated ReCoAlign objectives."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor, nn
from torch.nn import functional as F

LOSS_NAMES = (
    "semantic_preservation",
    "structural_consistency",
    "reasoning_alignment",
)


def semantic_preservation_loss(
    structure_summary: Tensor,
    visual_summary: Tensor,
    targets: Tensor | None = None,
) -> Tensor:
    """Keep object/attribute semantics in the latent interface.

    Without labels, normalized visual features provide a weak self-distillation
    target.  If a target representation is available, it is used as an additional
    supervised target inside the same semantic-preservation objective.
    """

    loss = 1.0 - F.cosine_similarity(structure_summary, visual_summary.detach(), dim=-1)
    result = loss.mean()
    if targets is None:
        return result
    target = targets.to(device=structure_summary.device, dtype=structure_summary.dtype)
    if target.shape == structure_summary.shape:
        result = result + F.mse_loss(structure_summary, target)
    elif target.ndim == 1 and structure_summary.ndim == 2:
        raise ValueError(
            "semantic_targets with class IDs require semantic logits; "
            "pass targets matching the structure summary for this objective"
        )
    else:
        raise ValueError("semantic_targets must match the structure summary shape")
    return result / 2.0


def structural_consistency_loss(
    logits: Tensor | Mapping[str, Tensor],
    targets: Tensor | Mapping[str, Tensor] | None,
) -> Tensor:
    """Use graph annotations as a training signal for object/relation structure."""

    if isinstance(logits, Mapping):
        if not logits:
            raise ValueError("structural logits mapping must not be empty")
        if targets is None:
            return next(iter(logits.values())).sum() * 0.0
        if not isinstance(targets, Mapping):
            if "relation" not in logits:
                raise ValueError("tensor structural targets require relation logits")
            return structural_consistency_loss(logits["relation"], targets)
        shared = sorted(set(logits) & set(targets))
        if not shared:
            raise ValueError("structural logits and targets have no shared supervision type")
        losses = [structural_consistency_loss(logits[name], targets[name]) for name in shared]
        return sum(losses) / len(losses)
    if isinstance(targets, Mapping):
        if "relation" not in targets:
            raise ValueError("tensor structural logits require relation targets")
        targets = targets["relation"]
    if targets is None:
        return logits.sum() * 0.0
    target = targets.to(device=logits.device)
    if target.shape == logits.shape and target.is_floating_point():
        return F.mse_loss(logits, target)
    if logits.ndim == 2 and target.ndim == 1:
        return F.cross_entropy(logits, target.long())
    if logits.ndim == 3 and target.ndim == 2:
        return F.cross_entropy(logits.transpose(1, 2), target.long())
    raise ValueError("structural_targets must be class IDs or match structural logits")


def reasoning_alignment_loss(logits: Tensor, labels: Tensor | None) -> Tensor:
    """Align the adapted context with the downstream reasoning answer."""

    if labels is None:
        return logits.sum() * 0.0
    target = labels.to(device=logits.device)
    if logits.ndim == 2 and target.ndim == 1:
        return F.cross_entropy(logits, target.long())
    if logits.ndim == 3 and target.ndim == 2:
        return F.cross_entropy(logits.transpose(1, 2), target.long())
    raise ValueError("answer_labels must be class IDs compatible with reasoning logits")


class ReCoAlignLosses(nn.Module):
    """Weighted container exposing exactly the three registered loss terms."""

    names = LOSS_NAMES

    def __init__(
        self,
        *,
        semantic_weight: float = 1.0,
        structural_weight: float = 1.0,
        reasoning_weight: float = 1.0,
    ) -> None:
        super().__init__()
        weights = (semantic_weight, structural_weight, reasoning_weight)
        if any(weight < 0 for weight in weights) or not any(weight > 0 for weight in weights):
            raise ValueError("loss weights must be non-negative and not all zero")
        self.weights = dict(zip(LOSS_NAMES, (float(weight) for weight in weights), strict=True))

    def forward(
        self,
        *,
        structure_summary: Tensor,
        visual_summary: Tensor,
        structural_logits: Tensor | Mapping[str, Tensor],
        reasoning_logits: Tensor,
        semantic_targets: Tensor | None = None,
        structural_targets: Tensor | Mapping[str, Tensor] | None = None,
        answer_labels: Tensor | None = None,
    ) -> dict[str, Tensor]:
        terms = {
            "semantic_preservation": semantic_preservation_loss(
                structure_summary, visual_summary, semantic_targets
            ),
            "structural_consistency": structural_consistency_loss(
                structural_logits, structural_targets
            ),
            "reasoning_alignment": reasoning_alignment_loss(reasoning_logits, answer_labels),
        }
        total = sum(terms[name] * self.weights[name] for name in LOSS_NAMES)
        return {**terms, "total": total}

    def to_dict(self) -> dict[str, object]:
        return {"weights": dict(self.weights), "names": list(LOSS_NAMES)}


__all__ = [
    "LOSS_NAMES",
    "ReCoAlignLosses",
    "reasoning_alignment_loss",
    "semantic_preservation_loss",
    "structural_consistency_loss",
]
