from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    root: Path
    source_phase0_dir: Path
    model_dir: Path
    seed: int = 20260818
    n_train: int = 720
    n_test: int = 288
    attention_samples: int = 24
    bootstrap_samples: int = 2000
    layer_indices: tuple[int, ...] = (0, 8, 16, 24, 32)
    vision_batch_size: int = 4
    protocol_mode: str = "full"
    force_vision: bool = False
    force_llm: bool = False
    device: str = "cuda"
    quantization: str = "nf4"
    cuda_decoder_layers: int = 28
    prompt_prefix: str = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        "USER: "
    )
    prompt_suffix: str = (
        "\nDescribe the two colored shapes and their spatial relation in one concise "
        "sentence. ASSISTANT:"
    )
    min_za_accuracy: float = 0.80
    max_object_drop: float = 0.10
    min_specific_drop: float = 0.15

    @property
    def source_dataset_dir(self) -> Path:
        return self.source_phase0_dir / "synthetic_dataset"

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs"

    @property
    def features_dir(self) -> Path:
        return self.root / "features"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    def validate(self) -> None:
        if self.n_train < 72 or self.n_test < 72:
            raise ValueError("n_train and n_test must each cover all 72 composition classes")
        if self.n_train % 72 or self.n_test % 72:
            raise ValueError("n_train and n_test must be divisible by 72")
        if not self.layer_indices or self.layer_indices[0] != 0:
            raise ValueError("layer_indices must begin at layer 0 (the Za identity boundary)")
        if tuple(sorted(set(self.layer_indices))) != self.layer_indices:
            raise ValueError("layer_indices must be unique and increasing")
        if self.layer_indices[-1] != 32:
            raise ValueError("LLaVA-1.5-7B protocol must include final hidden layer 32")
        if self.attention_samples < 1 or self.attention_samples > self.n_test:
            raise ValueError("attention_samples must be between 1 and n_test")
        if self.attention_samples % 4:
            raise ValueError("attention_samples must be divisible by four relations")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if self.quantization not in {"nf4", "none"}:
            raise ValueError("quantization must be 'nf4' or 'none'")
        if not 1 <= self.cuda_decoder_layers <= 32:
            raise ValueError("cuda_decoder_layers must be in [1, 32]")

    def scientific_protocol_valid(self) -> bool:
        return (
            self.protocol_mode == "full"
            and self.n_train >= 720
            and self.n_test >= 288
            and self.attention_samples >= 24
            and self.bootstrap_samples >= 2000
            and self.layer_indices == (0, 8, 16, 24, 32)
            and self.quantization == "nf4"
            and self.cuda_decoder_layers == 28
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("root", "source_phase0_dir", "model_dir"):
            result[key] = str(Path(result[key]).resolve())
        result["layer_indices"] = list(self.layer_indices)
        result["scientific_protocol_valid"] = self.scientific_protocol_valid()
        return result

    @classmethod
    def from_json(cls, path: Path, *, root: Path | None = None) -> ExperimentConfig:
        payload = json.loads(path.read_text(encoding="utf-8"))
        base = path.resolve().parent.parent
        payload["root"] = root.resolve() if root else base
        for key in ("source_phase0_dir", "model_dir"):
            value = Path(payload[key])
            payload[key] = value if value.is_absolute() else (base / value).resolve()
        payload["layer_indices"] = tuple(payload["layer_indices"])
        return cls(**payload)
