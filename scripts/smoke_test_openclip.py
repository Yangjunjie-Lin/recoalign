#!/usr/bin/env python3
"""Load an OpenCLIP checkpoint and encode two short captions."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch

    from recoalign.models.openclip_encoder import OpenCLIPConfig, OpenCLIPEncoder

    encoder = OpenCLIPEncoder(
        OpenCLIPConfig(
            model_name=args.model,
            pretrained=args.pretrained,
            device=args.device,
        )
    )
    features = encoder.encode_texts(
        [
            "a red cup to the left of a blue box",
            "a blue cup to the left of a red box",
        ]
    )

    print(f"Loaded: {args.model} / {args.pretrained}")
    print(f"Device: {features.device}")
    print(f"Embedding shape: {tuple(features.shape)}")
    print(f"Finite: {bool(torch.isfinite(features).all())}")
    if torch.cuda.is_available():
        print(f"Allocated VRAM: {torch.cuda.memory_allocated() / 1024**2:.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
