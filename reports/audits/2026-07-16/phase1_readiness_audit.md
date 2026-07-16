# Phase 1.2 Baseline Execution Readiness Audit

Audit date: 2026-07-16

This is an execution-readiness audit. It is not a baseline result report.

## 1. Executive summary

The locally audited WSL2 software and CUDA environment can load the target OpenCLIP model and run
inference. Real baseline execution has not started because all six authorized benchmark snapshots
and formal dataset manifests are missing. All 18 planned model-benchmark runs have
`execution_status = BLOCKED_MISSING_DATA`, `run_status = NOT_RUN`, and
`reportability_status = NOT_RUN`.

This branch only hardens repository hygiene, records the pre-execution readiness state, and makes
Windows diagnostic output tolerant of restricted console encodings. It contains no real benchmark
metric, prediction, embedding cache, cache/no-cache comparison, complete run, reportable run, or
locally reproduced evidence.

OpenCLIP CUDA smoke test confirms environment/model inference readiness only. It is not a benchmark
run and must not be counted as locally reproduced evidence.

## 2. Repository state

Repository-state observations below were captured at the start of this report revision. They are
operational metadata, not permanent experimental conclusions.

- Audit branch: `experiment/phase-1-real-baselines`.
- Branch pushed: yes.
- Audit snapshot commit: `db2e799d48a6b68082a0d4b2e9c9381b8bb64d89`.
- `origin/main`: `9249c5cbee9213400fdb6ae36844202368649282`.
- `origin/develop`: `0160d3cdb844312e555f6ec2fb1509754bd4d632`.
- Snapshot versus `origin/main`: 1 ahead, 0 behind.
- Snapshot versus `origin/develop`: 2 ahead, 0 behind.
- Pull request: not created at this snapshot; the public GitHub API returned an empty result.
- Merge to `develop`: no.
- Merge to `main`: no.

The final branch commit and live pull-request checks must be read from Git after this report revision
is committed; the snapshot SHA above deliberately does not attempt to self-reference that future
commit.

## 3. Environment

These values were observed locally in WSL2. They were not produced by GitHub Actions.

| Item | Local observation |
| --- | --- |
| Host | Windows 11 |
| Execution target | WSL2 Ubuntu, Jammy/22.04 package sources |
| Python | 3.10.12 |
| pip | 26.1.2 |
| PyTorch | 2.13.0+cu126 |
| OpenCLIP | 3.3.0 |
| CUDA available | `True` |
| PyTorch CUDA runtime | 12.6 |
| NVIDIA driver | 566.07 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| Total VRAM | 6,441,926,656 bytes / 6144 MiB |
| Smoke embedding | `(2, 512)`, all finite |
| Smoke peak allocated memory | 621,345,280 bytes / 592.6 MiB |
| Smoke peak reserved memory | 666,894,336 bytes / 636.0 MiB |

`pip check` reported no broken requirements. One PyTorch index timeout recovered automatically.
OpenCLIP warned that Hugging Face requests were unauthenticated, which affects rate limits but did
not prevent checkpoint loading. The Windows console-encoding issue found during the initial audit is
covered by a GPU-independent CP1252 regression test in this revision; WSL2 CUDA detection remains
unchanged.

## 4. Dataset readiness

| Dataset | Available | Formal manifest | Image inventory | Image hashes | Source reviewed | License reviewed | Verification status | Blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Flickr30K | missing | missing | missing | NOT_RUN | no | no | BLOCKED_MISSING_DATA | root, source annotation, normalized annotation, images, manifest, and inventory missing |
| MS COCO | missing | missing | missing | NOT_RUN | no | no | BLOCKED_MISSING_DATA | root, source annotation, normalized annotation, images, manifest, and inventory missing |
| SugarCrepe | missing | missing | missing | NOT_RUN | no | no | BLOCKED_MISSING_DATA | root, seven source files, normalized annotation, images, manifest, and inventory missing |
| ARO | missing | missing | missing | NOT_RUN | no | no | BLOCKED_MISSING_DATA | root, source JSONL, normalized annotation, images, manifest, and inventory missing |
| Winoground | missing | missing | missing | NOT_RUN | no | no | BLOCKED_MISSING_DATA | root, source JSONL, normalized annotation, images, manifest, and inventory missing |
| BiVLC | missing | missing | missing | NOT_RUN | no | no | BLOCKED_MISSING_DATA | root, source JSONL, normalized annotation, images, manifest, and inventory missing |

Only `*.example.yaml` planning templates exist. They are not formal manifests and do not establish
file sizes, SHA-256 values, inventory hashes, sample counts, source review, or license review. Each
formal-manifest verification attempt returned exit code 2 with `manifest does not exist`.

## 5. Test evidence

### Local evidence

| Check | Status | Evidence |
| --- | --- | --- |
| CLI help | COMPLETE | expected preparation, verification, run, and lifecycle commands listed |
| Baseline configuration validation | COMPLETE | 18/18 baseline configs validated after this revision |
| CI configuration validation | COMPLETE | 1/1 CI config validated after this revision |
| Ruff after this revision | COMPLETE | all checks passed |
| pytest after this revision | COMPLETE | 46 tests passed in 46.20 seconds |
| OpenCLIP CUDA smoke test | COMPLETE | `(2, 512)` finite text embeddings |
| Windows CP1252 regression before fix | FAILED | target test reproduced the missing safe-output helper |
| Windows CP1252 regression after fix | COMPLETE | unencodable localized text replaced without traceback |
| WSL2 environment diagnostic after fix | COMPLETE | CUDA, GPU, VRAM, PyTorch, and OpenCLIP still reported |

### GitHub Actions evidence

`NOT_RUN` for this readiness revision at the snapshot above because no pull request existed and the
workflow only triggers for pull requests or pushes to `main`. Local results are not represented as
CI results.

### Blocked checks

Dataset verification, all real runs, provenance inspection, prediction review, metric consistency,
and cache/no-cache comparisons are `BLOCKED_MISSING_DATA`.

Winoground caption token-multiset validation currently uses lowercase plus whitespace splitting.
That normalization limitation remains documented; no real Winoground records were available.

## 6. Phase 1 execution status

| Model | Benchmark | Execution status | Run status | Reportability status |
| --- | --- | --- | --- | --- |
| `openai_vit_b32` | Flickr30K | BLOCKED_MISSING_DATA | NOT_RUN | NOT_RUN |
| `openai_vit_b32` | MS COCO | BLOCKED_MISSING_DATA | NOT_RUN | NOT_RUN |
| `openai_vit_b32` | SugarCrepe | BLOCKED_MISSING_DATA | NOT_RUN | NOT_RUN |
| `openai_vit_b32` | ARO | BLOCKED_MISSING_DATA | NOT_RUN | NOT_RUN |
| `openai_vit_b32` | Winoground | BLOCKED_MISSING_DATA | NOT_RUN | NOT_RUN |
| `openai_vit_b32` | BiVLC | BLOCKED_MISSING_DATA | NOT_RUN | NOT_RUN |
| `openclip_vit_b32_laion2b` | Flickr30K | BLOCKED_MISSING_DATA | NOT_RUN | NOT_RUN |
| `openclip_vit_b32_laion2b` | MS COCO | BLOCKED_MISSING_DATA | NOT_RUN | NOT_RUN |
| `openclip_vit_b32_laion2b` | SugarCrepe | BLOCKED_MISSING_DATA | NOT_RUN | NOT_RUN |
| `openclip_vit_b32_laion2b` | ARO | BLOCKED_MISSING_DATA | NOT_RUN | NOT_RUN |
| `openclip_vit_b32_laion2b` | Winoground | BLOCKED_MISSING_DATA | NOT_RUN | NOT_RUN |
| `openclip_vit_b32_laion2b` | BiVLC | BLOCKED_MISSING_DATA | NOT_RUN | NOT_RUN |
| `openclip_vit_b16_laion2b` | Flickr30K | BLOCKED_MISSING_DATA | NOT_RUN | NOT_RUN |
| `openclip_vit_b16_laion2b` | MS COCO | BLOCKED_MISSING_DATA | NOT_RUN | NOT_RUN |
| `openclip_vit_b16_laion2b` | SugarCrepe | BLOCKED_MISSING_DATA | NOT_RUN | NOT_RUN |
| `openclip_vit_b16_laion2b` | ARO | BLOCKED_MISSING_DATA | NOT_RUN | NOT_RUN |
| `openclip_vit_b16_laion2b` | Winoground | BLOCKED_MISSING_DATA | NOT_RUN | NOT_RUN |
| `openclip_vit_b16_laion2b` | BiVLC | BLOCKED_MISSING_DATA | NOT_RUN | NOT_RUN |

The canonical machine-readable matrix is `phase1_execution_status.csv`. It contains status markers,
not numeric metrics or literature values. No row is `COMPLETE`, `REPORTABLE`, `FAILED`, or
`LOCALLY_REPRODUCED`.

## 7. Prediction readiness

`phase1_prediction_review_template.csv` is an empty template and contains no reviewed prediction
samples. It has one header row and zero sample rows. Prediction indexing, margins, ties,
directionality failures, 2x2 score conventions, and visual review are all `NOT_RUN`.

## 8. Remaining blockers

Prepare data only from authorized upstream copies. Replace every placeholder below with reviewed,
factual source and license text before execution.

```bash
recoalign prepare-flickr30k --karpathy-json data/flickr30k/incoming/dataset_flickr30k.json --dataset-root data/flickr30k --manifest-output manifests/datasets/flickr30k.yaml --source "REPLACE WITH VERIFIED SOURCE" --license "REPLACE WITH VERIFIED LICENSE NOTE" --hash-images
recoalign prepare-coco --karpathy-json data/mscoco/incoming/dataset_coco.json --dataset-root data/mscoco --manifest-output manifests/datasets/mscoco.yaml --source "REPLACE WITH VERIFIED SOURCE" --license "REPLACE WITH VERIFIED LICENSE NOTE" --hash-images
recoalign prepare-sugarcrepe --official-data-dir data/sugarcrepe/incoming --dataset-root data/sugarcrepe --manifest-output manifests/datasets/sugarcrepe.yaml --source "Official RAIVNLab SugarCrepe release, verified locally" --license "REPLACE WITH VERIFIED SUGARCREPE AND COCO LICENSE NOTE" --hash-images
recoalign prepare-aro --source-jsonl data/aro/incoming/aro.jsonl --dataset-root data/aro --manifest-output manifests/datasets/aro.yaml --source "REPLACE WITH VERIFIED OFFICIAL ARO EXPORT" --license "REPLACE WITH VERIFIED UPSTREAM TERMS" --hash-images
recoalign prepare-winoground --source-jsonl data/winoground/incoming/winoground.jsonl --dataset-root data/winoground --manifest-output manifests/datasets/winoground.yaml --source "REPLACE WITH VERIFIED OFFICIAL WINOGROUND EXPORT" --license "REPLACE WITH VERIFIED UPSTREAM TERMS" --hash-images
recoalign prepare-bivlc --source-jsonl data/bivlc/incoming/bivlc.jsonl --dataset-root data/bivlc --manifest-output manifests/datasets/bivlc.yaml --source "REPLACE WITH VERIFIED HUMAN-FILTERED BIVLC EXPORT" --license "REPLACE WITH VERIFIED UPSTREAM TERMS" --hash-images
```

The exact missing paths and verification errors are recorded in `phase1_data_blockers.csv`.

## 9. Recommended next action

1. Acquire and prepare authorized benchmark data.
2. Verify all six formal, hashed manifests.
3. Run the OpenCLIP ViT-B/32 LAION-2B six-benchmark gate sequentially.

Do not begin ReCoAlign training or the remaining matrix until that gate is audited.
