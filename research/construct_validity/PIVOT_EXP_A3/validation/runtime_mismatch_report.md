# PIVOT_EXP_A3 runtime mismatch report

Captured on 2026-08-24 before package installation, model loading, or A3 validation inference.

## Classification

Primary failure code: `CPU_ONLY_TORCH`.

The active shell resolves `python` to the system-level CPython 3.12.6 installation at
`C:\Users\Yangjunjie Lin\AppData\Local\Programs\Python\Python312\python.exe`. This is the
required Python release, but it is not an isolated or confirmed PIVOT_EXP_A2 environment. Its
installed Torch is `2.11.0+cpu`; `torch.version.cuda` and cuDNN are both `None`, and
`torch.cuda.is_available()` is false with zero CUDA devices.

The NVIDIA driver is operational. `nvidia-smi` reports driver 566.07, CUDA compatibility 12.7,
and an NVIDIA GeForce RTX 3060 Laptop GPU with compute capability 8.6. The failure is therefore
not `CUDA_DRIVER_UNAVAILABLE`.

## Frozen-baseline mismatch

| Component | Active interpreter | Frozen A2 baseline | Assessment |
|---|---:|---:|---|
| Python | 3.12.6 | 3.12.6 | version matches; interpreter provenance unconfirmed |
| PyTorch | 2.11.0+cpu | 2.7.1+cu126 | CPU build and version mismatch |
| Transformers | 5.14.1 | 4.52.4 | version mismatch |
| bitsandbytes | 0.50.1 | 0.48.2 | version mismatch |
| CUDA build in Torch | none | 12.6 | missing from active Torch |
| NVIDIA GPU | RTX 3060 Laptop GPU | RTX 3060 Laptop GPU | hardware identity matches |

## Scientific boundary

No A3 inference was started, no model weights were loaded, no package was installed or changed,
and no scientific status was assigned. The lifecycle remains `preflight_blocked_runtime`; the
correct operational outcome is `RUNTIME_BLOCKED_PREINFERENCE` until a frozen CUDA/NF4 interpreter
passes all runtime gates.

## Historical environment recovery

The original isolated environment was found at
`D:\UoM_CS\GitHub\recoalign\.venv\Scripts\python.exe`. Its creation time is 2026-08-18, and its
Python, Torch, CUDA, cuDNN, GPU, Transformers, and bitsandbytes identities match the frozen A2
provenance. No package in this environment was installed, upgraded, downgraded, or removed.

This recovered environment passes CUDA tensor, bitsandbytes CUDA backend, NF4 linear-layer,
checkpoint-integrity, weight-load, vision-tower, projector, forced-choice, M1 text-table, and M2
visual-legend smoke gates. It does not yet authorize A3 validation inference: the deterministic
development-only free-generation smoke returned the bare string `2`, which the frozen secondary
grammar correctly rejects. The specific remaining hard-gate failure is
`DEVELOPMENT_FREE_GENERATION_CONTRACT_SMOKE_FAILED`, and the operational outcome remains
`RUNTIME_BLOCKED_PREINFERENCE`, not a scientific `INCONCLUSIVE` or construct-validity decision.
