# Milestone A readiness ledger

Audit date: 2026-07-17

## Scope

This ledger records A0 readiness and the entry conditions for the first real Winoground closure.
It does not claim a reportable benchmark result, an external-method reproduction, or any
ReCoAlign training result.

## Release state

- PR #11, `Harden Winoground reproducibility gates`, was merged into `develop` as
  `671a06e07309b667d8e324e7ae852a50281a4e4f`.
- PR #12 merged that release into `main` as
  `9acc520544e0b8a0f1017a4e48196bbfa11e2980`.
- `origin/develop` is an ancestor of `origin/main`; the one-commit difference is the PR #12 merge
  commit.
- A duplicate release PR is therefore neither needed nor opened.
- The A1 work branch starts from the released `origin/main` commit above.

## Repository inventory

| Item | Status | Evidence |
| --- | --- | --- |
| Baseline configs | READY | exactly 18 YAML files: 3 encoders x 6 benchmarks |
| Dataset manifests | PARTIAL | Winoground has a historical hashed snapshot but pinned provenance regeneration is required; the other five formal manifests are absent |
| Checkpoint manifests | READY_FOR_REGISTRY_RESOLUTION | all three frozen encoders have committed registry manifests |
| Environment profiles | READY | WSL2 CUDA 12.6 / Python 3.10 profile is committed |
| Reports | READY | prior audit and Winoground evidence templates are present |
| Results | EMPTY_FOR_REAL_CLAIMS | no real reportable result is committed |
| Scripts | READY | environment, OpenCLIP smoke, Winoground export, audit, and comparison scripts are present |

## Environment preflight

The repaired `recoalign` conda environment passed `scripts/check_environment.py` with:

- Python 3.10.20;
- PyTorch 2.11.0+cu126;
- CUDA runtime 12.6;
- NVIDIA driver 566.07;
- NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB;
- OpenCLIP 3.3.0;
- NumPy 2.2.6;
- Pillow 12.2.0.

The OpenCLIP `ViT-B-32 / laion2b_s34b_b79k` smoke test passed in offline mode against a previously
downloaded official Hugging Face cache artifact. It produced two finite 512-dimensional
embeddings. The cached checkpoint artifact is 605,143,316 bytes with SHA-256
`ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6`.
The official model repository revision is `1a25a446712ba5ee05982a381eed697ef9b435cf`;
its repository metadata declares the MIT license. The committed lightweight checkpoint manifest
declares the local ignored artifact, so run creation verifies bytes and SHA-256 instead of recording
an unverified registry-only resolution. No checkpoint file is stored in Git.

The environment was repaired after discovering a truncated zero-byte CUDA library. Because the
committed bootstrap profile does not pin the PyTorch release line, the local repair explicitly
aligned torch 2.11.0, torchvision 0.26.0, and torchaudio 2.11.0 on CUDA 12.6. This is an environment
repair only; no frozen experiment config changed.

## Winoground entry state

- An authorized cached official dataset snapshot exists locally with 400 examples and 800 images.
- The cached official Hugging Face dataset ref is the 40-character commit
  `b400e173549071916ad1b3d449293bc8d8b4b763`.
- Live Hugging Face API calls currently fail with connection resets, so remote identity and login
  revalidation remain a recorded network blocker.
- The committed Winoground manifest is not formally usable: it has null revision and acquisition
  time and is explicitly marked `requires_regeneration_from_pinned_revision`.
- The local historical runs under the ignored output directory are not eligible evidence. They
  were produced from an older commit and the unpinned manifest.

## Matrix readiness

The config-to-matrix mapping is exactly 18 unique combinations. No canonical or verification run
ID is allocated in `run_registry.csv` before a run actually exists.

| Benchmark | openai_vit_b32 | openclip_vit_b32_laion2b | openclip_vit_b16_laion2b |
| --- | --- | --- | --- |
| Winoground | config present | config present; first A1 target | config present |
| ARO | config present | config present | config present |
| BiVLC | config present | config present | config present |
| Flickr30K | config present | config present | config present |
| MS COCO | config present | config present | config present |
| SugarCrepe | config present | config present | config present |

## Invalidated local results

All pre-existing Winoground output directories are invalid for Milestone A because they were
created before the reportability-gate release and/or against the unpinned manifest. Their metrics,
predictions, audits, and partial review material must not be promoted, copied into the registry, or
used as no-cache verification evidence. A1 requires new run IDs from one clean released commit
after formal manifest regeneration.

## A0 decision

A0 release and repository readiness are complete. A1 execution is permitted only after:

1. a pinned formal Winoground export and manifest verify successfully;
2. the manifest is committed;
3. the branch is clean at one frozen commit;
4. the canonical and cache-disabled verification runs are both newly executed from that commit.
