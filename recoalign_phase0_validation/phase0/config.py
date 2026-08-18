from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    root: Path
    seed: int = 20260818
    n_train: int = 8000
    n_test: int = 2000
    n_control_pairs: int = 500
    image_size: int = 224
    backend: str = "auto"
    model_name: str = "openai/clip-vit-base-patch32"
    llava_repo: str = "liuhaotian/llava-v1.5-7b"
    llava_vision_model: str = "openai/clip-vit-large-patch14-336"
    batch_size: int = 16
    device: str = "auto"
    probe_max_iter: int = 500
    bootstrap_samples: int = 2000
    force_data: bool = False
    force_features: bool = False
    protocol_mode: str = "full"

    @property
    def dataset_dir(self) -> Path:
        return self.root / "synthetic_dataset"

    @property
    def features_dir(self) -> Path:
        return self.root / "features"

    @property
    def results_dir(self) -> Path:
        return self.root / "experiment_results"

    def validate(self) -> None:
        # The full factorial contains 72 composition classes. Both partitions must
        # contain every class so that composition accuracy has a defined meaning.
        if self.n_train < 72 or self.n_test < 72:
            raise ValueError("n_train and n_test must each be at least 72")
        if self.n_control_pairs < 4:
            raise ValueError("n_control_pairs must be at least 4")
        if self.image_size < 128:
            raise ValueError("image_size must be at least 128")
        if self.batch_size < 1 or self.bootstrap_samples < 20:
            raise ValueError("batch_size must be positive and bootstrap_samples >= 20")
        if self.backend not in {"auto", "llava", "clip"}:
            raise ValueError("backend must be one of: auto, llava, clip")

    def scientific_protocol_valid(self) -> bool:
        return (
            self.protocol_mode == "full"
            and self.n_train >= 8000
            and self.n_test >= 2000
            and self.n_control_pairs >= 500
            and self.bootstrap_samples >= 2000
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["root"] = str(self.root.resolve())
        result["scientific_protocol_valid"] = self.scientific_protocol_valid()
        return result
