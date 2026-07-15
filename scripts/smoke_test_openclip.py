#!/usr/bin/env python3
"""Load an OpenCLIP checkpoint and encode two short captions."""

from __future__ import annotations

import argparse

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="amp", choices=["fp32", "amp", "amp_bf16"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from recoalign.models.openclip_encoder import OpenCLIPConfig, OpenCLIPEncoder

    encoder = OpenCLIPEncoder(
        OpenCLIPConfig(
            model_name=args.model,
            pretrained=args.pretrained,
            device=args.device,
            precision=args.precision,
        )
    )
    features = encoder.encode_texts(
        [
            "a red cup to the left of a blue box",
            "a blue cup to the left of a red box",
        ],
        batch_size=2,
    )
    print(f"Loaded: {args.model} / {args.pretrained}")
    print(f"Fingerprint: {encoder.fingerprint}")
    print(f"Embedding shape: {features.shape}")
    print(f"Finite: {bool(np.isfinite(features).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
