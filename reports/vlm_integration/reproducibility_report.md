# VLM Reproducibility Report

## LLaVA-1.5 dry-run identity

- Registry key: `llava_1_5_7b`
- Upstream identifier: `liuhaotian/llava-v1.5-7b`
- Revision: `4481d270cc22fd5c4d1bb5df129622006ccd9234`
- Local checkpoint: `outputs/models/llava-v1.5-7b`
- Checkpoint aggregate fingerprint: `6334826897d616bad74eeec6a84cb43d62f5f7c3957628d73ddfaa5734e6c5c6`
- Manifest SHA-256: `8f6ef275d87bc1051c8b38c5e1cee2c0ac7357103f4186588082f849e81c54d7`
- Verified files: 9/9 present, size matched, and content SHA-256 registered
- Checkpoint format / loader: `legacy_llava` / `legacy_transformers` (compatible)
- Processor: `LlavaProcessor` with `LlamaTokenizer` and `CLIPImageProcessor`
- Required runtime modules: torch, transformers, accelerate, sentencepiece, and bitsandbytes present
- Dtype / quantization: float16 / NF4
- Device mapping: auto
- Decoding: sampling disabled, temperature 0, one beam, maximum 32 new tokens
- Prompt: `reasoning-default-v1`
- Prompt file SHA-256: `1499f80e3c10d9cf2aaa2acdff143c0f77a9e76d18501cfa0a4e0b5aec4d7492`
- Dry-run state: `runtime_ready=true`, `hardware_ready=true`, `weights_loaded=false`, 0 predictions

## Real-weight execution attempt

- Environment: CUDA v12.6, PyTorch 2.7.1+cu126, NVIDIA GeForce RTX 3060 Laptop GPU
- Loader state: weights, CLIP vision tower, and multimodal projector loaded successfully
- Single-sample smoke: 3/3 conditions generated and normalized correctly
- EXP001 seed: `20260818` (registered seed, single-seed non-claim attempt)
- Deterministic cache entries completed: 629 of the expected 1,600 requests
- Final formal prediction/metric bundle: not written
- Termination: one generation made no cache progress for >6 minutes under a fully occupied 6GB
  device; process was safely interrupted and the incomplete directory was archived
- Evidence status: inconclusive; no accuracy, generalization, or interface-gap claim is permitted

## Captured runtime

- Python: 3.12.6
- PyTorch: 2.7.1+cu126
- CUDA runtime: 12.6
- cuDNN: 90701
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- GPU memory: 6,441,926,656 bytes
- Compute capability: 8.6

Every real evaluation additionally records the dataset manifest/hash, model config/hash, checkpoint
manifest and file verification, generation configuration, prompt version/hash, git metadata, cache
state, batch size, and complete predictions. The authoritative dry-run record is under
`outputs/EXP001/llava_1_5_7b/run.json`.
