"""Scientific stage definitions and trainable-parameter policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from torch import nn


class TrainingStage(str, Enum):
    STRUCTURED_INTERFACE_PRETRAINING = "structured_interface_pretraining"
    REASONING_ALIGNMENT = "reasoning_alignment"
    INSTRUCTION_ADAPTATION = "instruction_adaptation"


class FreezePolicy(str, Enum):
    INTERFACE_ONLY = "interface_only"
    INTERFACE_AND_PROJECTOR = "interface_and_projector"
    FULL_FINETUNING = "full_finetuning"


@dataclass(frozen=True)
class StageDefinition:
    stage: TrainingStage
    scientific_goal: str
    required_supervision: tuple[str, ...]
    graph_at_inference: bool = False


STAGE_DEFINITIONS = {
    TrainingStage.STRUCTURED_INTERFACE_PRETRAINING: StageDefinition(
        TrainingStage.STRUCTURED_INTERFACE_PRETRAINING,
        "learn visual-to-structure latent representations",
        ("semantic_targets", "structural_targets"),
    ),
    TrainingStage.REASONING_ALIGNMENT: StageDefinition(
        TrainingStage.REASONING_ALIGNMENT,
        "align learned structure tokens with answer reasoning",
        ("answer_labels",),
    ),
    TrainingStage.INSTRUCTION_ADAPTATION: StageDefinition(
        TrainingStage.INSTRUCTION_ADAPTATION,
        "adapt the same interface to preregistered real-task instructions",
        ("answer_labels",),
    ),
}


def parse_stage(value: str | TrainingStage) -> TrainingStage:
    if isinstance(value, TrainingStage):
        return value
    aliases = {
        "stage1": TrainingStage.STRUCTURED_INTERFACE_PRETRAINING,
        "stage2": TrainingStage.REASONING_ALIGNMENT,
        "stage3": TrainingStage.INSTRUCTION_ADAPTATION,
    }
    try:
        return aliases.get(value, TrainingStage(value))
    except ValueError as exc:
        raise ValueError(f"unsupported ReCoAlign training stage: {value}") from exc


def parse_freeze_policy(value: str | FreezePolicy) -> FreezePolicy:
    if isinstance(value, FreezePolicy):
        return value
    try:
        return FreezePolicy(value)
    except ValueError as exc:
        choices = ", ".join(policy.value for policy in FreezePolicy)
        raise ValueError(f"unsupported freeze policy {value!r}; expected one of {choices}") from exc


def apply_freeze_policy(model: nn.Module, policy: str | FreezePolicy) -> dict[str, object]:
    """Apply an auditable policy to a core model or real-backbone wrapper.

    A bare :class:`ReCoAlignModel` contains only interface-side modules, so both
    interface policies train its interface/projectors/heads.  On a real wrapper,
    parameters named ``vision_encoder`` or ``llm`` remain frozen unless full
    fine-tuning is explicitly selected.
    """

    selected = parse_freeze_policy(policy)
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    for name, parameter in model.named_parameters():
        if selected is FreezePolicy.FULL_FINETUNING:
            trainable = True
        else:
            interface = name.startswith(
                ("interface.", "alignment.", "structural_heads.", "reasoning_head.")
            ) or any(
                marker in name
                for marker in (
                    ".interface.",
                    ".alignment.",
                    ".structural_heads.",
                    ".reasoning_head.",
                )
            )
            projector = name.startswith("projector.") or ".projector." in name
            trainable = interface or (
                selected is FreezePolicy.INTERFACE_AND_PROJECTOR and projector
            )
        parameter.requires_grad_(trainable)
        (trainable_names if trainable else frozen_names).append(name)
    if not trainable_names:
        raise ValueError(f"freeze policy {selected.value} leaves no trainable parameters")
    return {
        "policy": selected.value,
        "trainable_parameter_tensors": len(trainable_names),
        "frozen_parameter_tensors": len(frozen_names),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "frozen_parameters": sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        ),
        "trainable_names": trainable_names,
        "frozen_names": frozen_names,
    }


__all__ = [
    "FreezePolicy",
    "STAGE_DEFINITIONS",
    "StageDefinition",
    "TrainingStage",
    "apply_freeze_policy",
    "parse_freeze_policy",
    "parse_stage",
]
