# Milestone A readiness ledger

Audit date: 2026-07-18

## Scope

This ledger records A0 readiness and the current pre-promotion state of the first real Winoground
closure. It does not claim a reportable benchmark result, an external-method reproduction, or any
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
| Dataset manifests | PARTIAL | Winoground has a pinned, locally verified formal manifest; the other five formal manifests are absent |
| Checkpoint manifests | PARTIAL | the A1 OpenCLIP B/32 LAION2B checkpoint bytes and SHA-256 are verified; the other two encoders remain registry-only |
| Environment profiles | READY | WSL2 CUDA 12.6 / Python 3.10 profile is committed |
| Reports | READY | prior audit and Winoground evidence templates are present |
| Results | MACHINE_COMPLETE_HUMAN_PENDING | the A1 canonical and independent no-cache run pair passed machine audit and comparison but is not promoted |
| Scripts | READY | environment, OpenCLIP smoke, export, strict audit, comparison, and local human-review scripts are present |

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

## Winoground A1 state

- An authorized official snapshot was exported locally with 400 examples and 800 images.
- The fixed official Hugging Face dataset revision is the 40-character commit
  `b400e173549071916ad1b3d449293bc8d8b4b763`.
- The committed formal manifest is `pinned_revision_verified`, declares all three formal files,
  includes the complete 800-image SHA-256 inventory, and passed local verification.
- The manifest SHA-256 is
  `65f142abb0d1d44845824e952f50befd6a6c017da8769f945fc4f35d74de8020`.
- Live Hugging Face API calls currently fail with connection resets, so remote identity and login
  revalidation remains a recorded network limitation; the pinned official cached export and local
  hash verification are unaffected.
- The cache-enabled canonical run
  `wg-openclip-b32-laion2b-canonical-20260717-13d2c51` is `complete` with `review: null`.
- The separate cache-disabled verification run
  `wg-openclip-b32-laion2b-verification-nocache-20260717-13d2c51` is `complete` with
  `review: null`.
- Both runs were produced from clean Git commit
  `13d2c51730ec8677f342dec880f9b57d2cb2abee`, the same resolved config, and the same
  dataset/checkpoint manifests.
- Independent audits recomputed all 400 decisions and 92 metrics, verified annotation alignment,
  and passed for both runs.
- Pre-promotion comparison passed every identity, cache, metric, prediction-order, score, and
  decision gate. Maximum metric and score absolute differences are both zero, and all decision
  difference counts are zero.
- `reports/experiments/winoground/reviewed_sample_ids.csv` is still a review queue: all 400 sample
  IDs and machine-generated groups align with canonical predictions, but 0 of 400 rows has complete
  human review fields.
- The canonical run is not promoted and no formal result record exists.

## Matrix readiness

The config-to-matrix mapping is exactly 18 unique combinations. The one actual A1 pair is recorded;
no run ID is allocated for a matrix cell before its run exists.

| Benchmark | openai_vit_b32 | openclip_vit_b32_laion2b | openclip_vit_b16_laion2b |
| --- | --- | --- | --- |
| Winoground | config present; not run | machine pair complete; human review and promotion pending | config present; not run |
| ARO | config present | config present | config present |
| BiVLC | config present | config present | config present |
| Flickr30K | config present | config present | config present |
| MS COCO | config present | config present | config present |
| SugarCrepe | config present | config present | config present |

## Invalidated local results

All Winoground outputs that predate the pinned formal manifest and reportability-gate release remain
invalid for Milestone A. The two run IDs recorded above replaced those historical outputs and are
the only eligible A1 machine evidence. The audit-CLI change made after the runs only exposes the
annotation-alignment check already enforced by package-level integrity validation; it does not
change evaluation logic, predictions, metrics, config, or manifests, so the recorded pair is not
invalidated.

## A0/A1 decision

A0 release and readiness are complete. A1 machine execution is complete. A1 promotion remains
blocked until:

1. one human reviewer actually inspects all 400 image-caption mappings;
2. all review fields pass `validate_prediction_review`;
3. the formal CLI promotes only the canonical run;
4. lightweight result evidence and registries are updated and the full repository validation
   passes.

A2 must not start before the A1 closure PR is reviewed and merged into `develop`.
