from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    # Some Windows proxy setups stall in the optional Xet transport for large shards.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    target = Path(__file__).resolve().parent.parent / "outputs" / "models" / "llava-v1.5-7b"
    result = snapshot_download(
        "liuhaotian/llava-v1.5-7b",
        local_dir=target,
        allow_patterns=("*.json", "*.model", "pytorch_model-*.bin", "mm_projector.bin"),
    )
    print(result)


if __name__ == "__main__":
    main()
