from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    root: Path
    source_phase0b_dir: Path
    model_dir: Path
    seed: int = 20260818
    n_train: int = 720
    n_test: int = 288
    attention_test_samples: int = 288
    bootstrap_samples: int = 2000
    layer_indices: tuple[int, ...] = (0, 8, 16, 24, 32)
    choice_words: tuple[str, ...] = ("left", "right", "above", "below")
    answer_prefix: str = " "
    candidate_scoring: str = "teacher_forced_shared_answer_prefix"
    attention_collection: str = "separate_eager_diagnostic_forward"
    probe_c: float = 1.0
    probe_max_iter: int = 1000
    patch_layers: tuple[int, ...] = (8, 16, 24)
    patch_alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    primary_availability_layer: int = 32
    primary_patch_layer: int = 16
    primary_patch_alpha: float = 1.0
    max_patch_failures: int = 24
    min_availability_accuracy: float = 0.80
    min_gap: float = 0.15
    min_error_trials: int = 24
    min_error_availability_accuracy: float = 0.75
    min_patch_recovery: float = 0.15
    min_patch_advantage: float = 0.10
    protocol_mode: str = "full"
    protocol_version: int = 3
    device: str = "cuda"
    quantization: str = "nf4"
    cuda_decoder_layers: int = 28
    prompt_prefix: str = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        "USER: "
    )

    @property
    def features_dir(self) -> Path:
        return self.root / "features"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs"

    @property
    def source_features_dir(self) -> Path:
        return self.source_phase0b_dir / "features"

    def validate(self) -> None:
        if self.n_train != 720 or self.n_test != 288:
            raise ValueError("The registered Phase 0-C split is exactly 720 train / 288 test")
        if self.attention_test_samples != self.n_test:
            raise ValueError("Attention must be collected for the complete held-out test set")
        if self.layer_indices != (0, 8, 16, 24, 32):
            raise ValueError("Registered hidden states are exactly 0/8/16/24/32")
        if self.choice_words != ("left", "right", "above", "below"):
            raise ValueError("The registered behavioral task is the four spatial relations")
        if self.answer_prefix != " ":
            raise ValueError("The registered answer prefix is the tokenizer's shared space token")
        if self.candidate_scoring != "teacher_forced_shared_answer_prefix":
            raise ValueError("Candidate scoring must occur after the shared answer prefix")
        if self.attention_collection != "separate_eager_diagnostic_forward":
            raise ValueError("Attention must be isolated from the behavioral forward pass")
        if self.patch_layers != (8, 16, 24):
            raise ValueError("Registered patch layers are exactly 8/16/24")
        if self.patch_alphas != (0.0, 0.25, 0.5, 0.75, 1.0):
            raise ValueError("Registered patch alphas changed")
        if self.primary_availability_layer != 32:
            raise ValueError("The primary availability layer is fixed at 32")
        if self.primary_patch_layer != 16 or self.primary_patch_alpha != 1.0:
            raise ValueError("The primary causal test is the full layer-16 patch")
        if self.bootstrap_samples < 2000 or self.max_patch_failures < 24:
            raise ValueError("The full protocol requires 2000 bootstraps and 24 patch failures")
        if self.quantization not in {"nf4", "none"}:
            raise ValueError("quantization must be nf4 or none")

    def scientific_protocol_valid(self) -> bool:
        return (
            self.protocol_mode == "full"
            and self.protocol_version == 3
            and self.n_train == 720
            and self.n_test == 288
            and self.attention_test_samples == 288
            and self.bootstrap_samples >= 2000
            and self.layer_indices == (0, 8, 16, 24, 32)
            and self.answer_prefix == " "
            and self.candidate_scoring == "teacher_forced_shared_answer_prefix"
            and self.attention_collection == "separate_eager_diagnostic_forward"
            and self.patch_layers == (8, 16, 24)
            and self.patch_alphas == (0.0, 0.25, 0.5, 0.75, 1.0)
            and self.primary_availability_layer == 32
            and self.primary_patch_layer == 16
            and self.quantization == "nf4"
            and self.cuda_decoder_layers == 28
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("root", "source_phase0b_dir", "model_dir"):
            payload[key] = str(Path(payload[key]).resolve())
        for key in ("layer_indices", "choice_words", "patch_layers", "patch_alphas"):
            payload[key] = list(payload[key])
        payload["scientific_protocol_valid"] = self.scientific_protocol_valid()
        return payload

    @classmethod
    def from_json(cls, path: Path, *, root: Path | None = None) -> "ExperimentConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        base = path.resolve().parent.parent
        payload["root"] = root.resolve() if root else base
        for key in ("source_phase0b_dir", "model_dir"):
            value = Path(payload[key])
            payload[key] = value if value.is_absolute() else (base / value).resolve()
        for key in ("layer_indices", "choice_words", "patch_layers", "patch_alphas"):
            payload[key] = tuple(payload[key])
        return cls(**payload)
