"""A small, inference-oriented OpenCLIP adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OpenCLIPConfig:
    """Configuration required to load an OpenCLIP checkpoint."""

    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    device: str = "cuda"


class OpenCLIPEncoder:
    """Load OpenCLIP and expose normalized image and text encoders.

    Heavy optional dependencies are imported lazily so metric-only utilities and CI can run
    without installing PyTorch or OpenCLIP.
    """

    def __init__(self, config: OpenCLIPConfig | None = None) -> None:
        self.config = config or OpenCLIPConfig()

        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "OpenCLIP support is not installed. Install PyTorch first, then run "
                "`pip install -e '.[openclip]'`."
            ) from exc

        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

        self._torch = torch
        self._open_clip = open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.config.model_name,
            pretrained=self.config.pretrained,
            device=self.config.device,
        )
        self.tokenizer = open_clip.get_tokenizer(self.config.model_name)
        self.model.eval()

    def encode_texts(self, texts: list[str]) -> Any:
        """Return L2-normalized text embeddings on the configured device."""
        torch = self._torch
        tokens = self.tokenizer(texts).to(self.config.device)
        with torch.inference_mode():
            features = self.model.encode_text(tokens)
            return torch.nn.functional.normalize(features, dim=-1)

    def encode_images(self, images: list[Any]) -> Any:
        """Preprocess PIL images and return L2-normalized image embeddings."""
        torch = self._torch
        batch = torch.stack([self.preprocess(image) for image in images]).to(self.config.device)
        with torch.inference_mode():
            features = self.model.encode_image(batch)
            return torch.nn.functional.normalize(features, dim=-1)
