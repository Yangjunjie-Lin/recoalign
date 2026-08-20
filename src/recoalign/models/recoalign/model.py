"""Trainable ReCoAlign model and its checkpoint contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from recoalign.models.interface import StructureAlignmentModule

from .interface import ReCoAlignInterface
from .losses import LOSS_NAMES, ReCoAlignLosses


@dataclass(frozen=True)
class ReCoAlignConfig:
    """Serializable architecture and objective configuration."""

    visual_dim: int = 16
    interface_dim: int = 32
    llm_dim: int = 32
    num_structure_tokens: int = 8
    num_heads: int = 4
    num_relation_classes: int = 8
    num_object_classes: int = 8
    num_attribute_classes: int = 8
    num_composition_classes: int = 8
    num_answer_classes: int = 4
    dropout: float = 0.0
    semantic_loss_weight: float = 1.0
    structural_loss_weight: float = 1.0
    reasoning_loss_weight: float = 1.0
    include_visual_tokens: bool = True
    typed_structure_tokens: bool = True
    use_structure_tokens: bool = True
    random_structure_tokens: bool = False

    def __post_init__(self) -> None:
        if min(
            self.visual_dim,
            self.interface_dim,
            self.llm_dim,
            self.num_structure_tokens,
            self.num_heads,
            self.num_relation_classes,
            self.num_object_classes,
            self.num_attribute_classes,
            self.num_composition_classes,
            self.num_answer_classes,
        ) <= 0:
            raise ValueError("all ReCoAlign dimensions and class counts must be positive")
        if self.interface_dim % self.num_heads:
            raise ValueError("interface_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if any(
            weight < 0
            for weight in (
                self.semantic_loss_weight,
                self.structural_loss_weight,
                self.reasoning_loss_weight,
            )
        ):
            raise ValueError("loss weights must be non-negative")
        if self.random_structure_tokens and not self.use_structure_tokens:
            raise ValueError("random_structure_tokens requires use_structure_tokens=true")

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> ReCoAlignConfig:
        """Build a config from a flat mapping or ``model:`` YAML section."""

        source = payload.get("model", payload)
        if not isinstance(source, dict):
            raise TypeError("ReCoAlign model config must be a mapping")
        aliases = {
            "num_tokens": "num_structure_tokens",
            "structure_tokens": "num_structure_tokens",
            "num_relations": "num_relation_classes",
            "num_answers": "num_answer_classes",
        }
        values = {aliases.get(key, key): value for key, value in source.items()}
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReCoAlignModel(nn.Module):
    """Learn visual semantics → structure tokens → reasoning context.

    ``forward`` accepts visual tokens only.  There is intentionally no graph input
    argument: graph annotations can enter only through ``compute_loss`` during
    training, which prevents oracle-graph prompting from being mistaken for the
    proposed method.
    """

    method_name = "recoalign"
    loss_names = LOSS_NAMES

    def __init__(self, config: ReCoAlignConfig | dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = (
            config
            if isinstance(config, ReCoAlignConfig)
            else ReCoAlignConfig.from_mapping(config or {})
        )
        cfg = self.config
        self.interface = ReCoAlignInterface(
            cfg.visual_dim,
            cfg.interface_dim,
            cfg.llm_dim,
            cfg.num_structure_tokens,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            include_visual_tokens=cfg.include_visual_tokens,
            typed_tokens=cfg.typed_structure_tokens,
        )
        self.alignment = StructureAlignmentModule(cfg.visual_dim, cfg.interface_dim)
        self.structural_heads = nn.ModuleDict(
            {
                "object": nn.Linear(cfg.interface_dim, cfg.num_object_classes),
                "attribute": nn.Linear(cfg.interface_dim, cfg.num_attribute_classes),
                "relation": nn.Linear(cfg.interface_dim, cfg.num_relation_classes),
                "composition": nn.Linear(cfg.interface_dim, cfg.num_composition_classes),
            }
        )
        self.reasoning_head = nn.Linear(cfg.llm_dim, cfg.num_answer_classes)
        self.reasoning_pool = nn.Linear(cfg.llm_dim, 1)
        self.losses = ReCoAlignLosses(
            semantic_weight=cfg.semantic_loss_weight,
            structural_weight=cfg.structural_loss_weight,
            reasoning_weight=cfg.reasoning_loss_weight,
        )

    @property
    def num_structure_tokens(self) -> int:
        return self.config.num_structure_tokens

    def extract_structure_tokens(self, visual_tokens: Tensor) -> Tensor:
        """Extract latent slots; this method never accepts graph annotations."""

        if not self.config.use_structure_tokens:
            batch = visual_tokens.shape[0]
            return visual_tokens.new_zeros(
                batch,
                self.config.num_structure_tokens,
                self.config.interface_dim,
            )
        if self.config.random_structure_tokens:
            return visual_tokens.new_empty(
                visual_tokens.shape[0],
                self.config.num_structure_tokens,
                self.config.interface_dim,
            ).normal_()
        return self.interface.extract_structure_tokens(visual_tokens)

    def forward(self, visual_tokens: Tensor) -> dict[str, Tensor]:
        if visual_tokens.ndim != 3 or visual_tokens.shape[-1] != self.config.visual_dim:
            raise ValueError(
                "visual_tokens must have shape [batch, tokens, visual_dim] "
                f"with visual_dim={self.config.visual_dim}"
            )
        structure_tokens = self.extract_structure_tokens(visual_tokens)
        return self.forward_with_structure_tokens(visual_tokens, structure_tokens)

    def forward_with_structure_tokens(
        self,
        visual_tokens: Tensor,
        structure_tokens: Tensor,
    ) -> dict[str, Tensor]:
        """Run the reasoning path with an explicit intervention tensor.

        This boundary is used only for causal ablations (remove/shuffle/replace).
        It still accepts learned-token tensors, never graph annotations.
        """
        if visual_tokens.ndim != 3 or visual_tokens.shape[-1] != self.config.visual_dim:
            raise ValueError(
                "visual_tokens must have shape [batch, tokens, visual_dim] "
                f"with visual_dim={self.config.visual_dim}"
            )
        expected = (
            visual_tokens.shape[0],
            self.config.num_structure_tokens,
            self.config.interface_dim,
        )
        if tuple(structure_tokens.shape) != expected:
            raise ValueError(f"structure_tokens must have shape {expected}")
        context = self.interface.adapter(visual_tokens, structure_tokens)
        visual_summary, structure_summary = self.alignment(visual_tokens, structure_tokens)
        structural_logits_by_type = {
            name: head(structure_summary) for name, head in self.structural_heads.items()
        }
        attention_logits = self.reasoning_pool(context).squeeze(-1)
        attention = torch.softmax(attention_logits, dim=1)
        reasoning_summary = (context * attention.unsqueeze(-1)).sum(dim=1)
        reasoning_logits = self.reasoning_head(reasoning_summary)
        return {
            "structure_tokens": structure_tokens,
            "context": context,
            "visual_summary": visual_summary,
            "structure_summary": structure_summary,
            "structural_logits": structural_logits_by_type["relation"],
            "structural_logits_by_type": structural_logits_by_type,
            "reasoning_logits": reasoning_logits,
            "reasoning_attention": attention,
        }

    def compute_loss(
        self,
        outputs: dict[str, Tensor],
        *,
        semantic_targets: Tensor | None = None,
        structural_targets: dict[str, Tensor] | None = None,
        object_targets: Tensor | None = None,
        attribute_targets: Tensor | None = None,
        relation_targets: Tensor | None = None,
        composition_targets: Tensor | None = None,
        answer_labels: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute the registered three-term objective from a forward pass."""

        targets = dict(structural_targets or {})
        for name, target in (
            ("object", object_targets),
            ("attribute", attribute_targets),
            ("relation", relation_targets),
            ("composition", composition_targets),
        ):
            if target is not None:
                targets[name] = target
        structural_logits = outputs.get("structural_logits_by_type", outputs["structural_logits"])
        return self.losses(
            structure_summary=outputs["structure_summary"],
            visual_summary=outputs["visual_summary"],
            structural_logits=structural_logits,
            reasoning_logits=outputs["reasoning_logits"],
            semantic_targets=semantic_targets,
            structural_targets=targets or relation_targets,
            answer_labels=answer_labels,
        )

    def loss(
        self,
        visual_tokens: Tensor,
        *,
        semantic_targets: Tensor | None = None,
        structural_targets: dict[str, Tensor] | None = None,
        object_targets: Tensor | None = None,
        attribute_targets: Tensor | None = None,
        relation_targets: Tensor | None = None,
        composition_targets: Tensor | None = None,
        answer_labels: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Convenience method for toy and downstream trainers."""

        return self.compute_loss(
            self(visual_tokens),
            semantic_targets=semantic_targets,
            structural_targets=structural_targets,
            object_targets=object_targets,
            attribute_targets=attribute_targets,
            relation_targets=relation_targets,
            composition_targets=composition_targets,
            answer_labels=answer_labels,
        )

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        step: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "method": self.method_name,
            "format_version": 1,
            "config": self.config.to_dict(),
            "state_dict": self.state_dict(),
            "losses": self.losses.to_dict(),
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if step is not None:
            payload["step"] = int(step)
        if extra:
            payload["extra"] = dict(extra)
        torch.save(payload, destination)
        return destination

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        payload = _torch_load(Path(path), map_location=map_location)
        if payload.get("method") != self.method_name:
            raise ValueError(f"checkpoint is not a {self.method_name} checkpoint")
        if payload.get("config") != self.config.to_dict():
            raise ValueError("checkpoint config does not match this ReCoAlignModel")
        _load_state_compatibly(self, payload["state_dict"])
        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        return payload

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> ReCoAlignModel:
        payload = _torch_load(Path(path), map_location=map_location)
        model = cls(ReCoAlignConfig.from_mapping(payload["config"]))
        _load_state_compatibly(model, payload["state_dict"])
        return model


def _torch_load(path: Path, *, map_location: str | torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"ReCoAlign checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 2.0 compatibility
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("ReCoAlign checkpoint must contain a mapping")
    return payload


def _load_state_compatibly(model: nn.Module, state_dict: dict[str, Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"reasoning_pool.weight", "reasoning_pool.bias"}
    if set(unexpected) or set(missing) - allowed_missing:
        raise ValueError(
            "checkpoint state does not match ReCoAlign model: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


__all__ = ["ReCoAlignConfig", "ReCoAlignModel"]
