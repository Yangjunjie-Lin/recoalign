# Phase 1.2 Real Baseline Audit

Audit date: 2026-07-16

## 1. Executive summary

The baseline software and CUDA inference environment are runnable, but no real benchmark is
currently runnable because all six authorized dataset snapshots and their formal manifests are
absent. Zero real runs were started: 0 complete, 0 reportable, 0 failed runs, and 18 planned matrix
runs blocked by missing data. No output directory, metrics file, prediction file, cache comparison,
or run provenance was generated.

One repository-safety defect was found and fixed: ARO, Winoground, and BiVLC data roots were not
Git-ignored. A regression test now covers all six real dataset roots. This change cannot affect
previous or future metric values. The project has not reached the prediction-based failure taxonomy
stage because there are no real predictions to inspect.

Acceptance status:

- PASS: local `develop` fast-forwarded to `origin/main`; experiment branch created.
- PASS: Ruff, 39 tests, 18/18 config validations, WSL2 CUDA check, and OpenCLIP smoke test.
- PASS: all six datasets individually audited and proven blocked by missing data.
- BLOCKED: real runs, provenance inspection, prediction/metric consistency checks, and three
  cache-free reruns require authorized benchmark data.
- PASS: no synthetic result was represented as a real result; no dataset, weight, cache, output,
  or large prediction artifact was added.

## 2. Repository state

- Current branch: `experiment/phase-1-real-baselines`.
- Current branch base: `9249c5cbee9213400fdb6ae36844202368649282`.
- `origin/main`: `9249c5cbee9213400fdb6ae36844202368649282`.
- `origin/develop`: `0160d3cdb844312e555f6ec2fb1509754bd4d632`.
- Before synchronization, `origin/develop...origin/main` was `0 1`; `origin/develop` was an ancestor
  of `origin/main`. Local `develop` was safely fast-forwarded to `origin/main`.
- The experiment branch was created from the synchronized local `develop`.
- Local commits created by this audit: none at the time this report was written.
- Push/PR: none. No merge to `develop` or `main` was performed.

## 3. Environment

| Item | Audited value | Status |
| --- | --- | --- |
| Host | Windows 11 | observed |
| Execution target | WSL2 `Ubuntu` (Jammy/22.04 package sources) | PASS |
| Python | 3.10.12 | PASS |
| pip | 26.1.2 | PASS |
| PyTorch | 2.13.0+cu126 | PASS |
| OpenCLIP | 3.3.0 | PASS |
| CUDA available | `True` | PASS |
| PyTorch CUDA runtime | 12.6 | PASS |
| NVIDIA driver | 566.07 | PASS |
| `nvidia-smi` maximum CUDA | 12.7 | observed |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU | PASS |
| Total VRAM | 6,441,926,656 bytes / 6144 MiB | PASS |
| Smoke embedding | `(2, 512)`, all finite | PASS |
| Smoke peak allocated VRAM | 621,345,280 bytes / 592.6 MiB | PASS |
| Smoke peak reserved VRAM | 666,894,336 bytes / 636.0 MiB | observed |

`pip check` reported no broken requirements. Installation encountered one transient read timeout
against the PyTorch wheel index and recovered automatically. OpenCLIP emitted the expected warning
that Hugging Face requests were unauthenticated; the checkpoint still loaded successfully. A native
Windows invocation of `scripts/check_environment.py` hit a CP1252/localized-error printing issue;
the documented WSL2 target invocation passed and is the audited execution environment.

## 4. Dataset audit

| Dataset | Available | Verified | Hashed | Expected constraint | License/source checked | Blocker |
| --- | --- | --- | --- | --- | --- | --- |
| Flickr30K | no | no; exit 2 | no | 1,000 images, 5,000 captions | no | root, source annotation, normalized annotation, images, formal manifest, inventory absent |
| MS COCO | no | no; exit 2 | no | 5,000 images, 25,000 captions | no | root, source annotation, normalized annotation, images, formal manifest, inventory absent |
| SugarCrepe | no | no; exit 2 | no | seven required categories; count not pinned in config | no | root, seven source JSON files, normalized annotation, images, formal manifest, inventory absent |
| ARO | no | no; exit 2 | no | four required subsets; count not pinned in config | no | root, source JSONL, normalized annotation, images, formal manifest, inventory absent |
| Winoground | no | no; exit 2 | no | 400 samples; caption token multiset must be 100% | no | root, source JSONL, normalized annotation, images, formal manifest, inventory absent |
| BiVLC | no | no; exit 2 | no | human-filtered protocol; count not pinned in config | no | root, source JSONL, normalized annotation, images, formal manifest, inventory absent |

Only `*.example.yaml` planning templates exist. They contain placeholders or empty `files` arrays and
are not formal experiment manifests. Consequently, manifest schema validity, declared file sizes,
SHA-256 values, per-image inventory hashes, sample counts, source terms, and license terms cannot be
verified. Every `verify-dataset` command failed with exit code 2 and the corresponding message:
`error: manifest does not exist: manifests/datasets/<dataset>.yaml`.

## 5. Tests

| Check | Result | Evidence |
| --- | --- | --- |
| CLI help | PASS | all preparation, verification, baseline, finalization, promotion, and table commands listed |
| Config validation | PASS | 18/18 `configs/baseline/*.yaml` validated |
| Ruff | PASS | `All checks passed!` |
| pytest before audit fix | PASS | 33 tests collected and passed |
| hygiene regression before fix | expected FAIL | ARO, Winoground, BiVLC roots returned `git check-ignore` exit 1 |
| hygiene regression after fix | PASS | six parameter cases passed |
| full pytest after fix | PASS | 39 tests collected and passed |
| CUDA environment check | PASS | CUDA available, GPU and VRAM reported |
| OpenCLIP CUDA smoke | PASS | checkpoint loaded; `(2, 512)` finite embeddings |
| Dataset verification | BLOCKED | six formal manifests absent; each command exited 2 |
| Cached baseline | NOT RUN | data preconditions failed |
| No-cache comparison | NOT RUN | cached baseline does not exist; data preconditions failed |

The Winoground token multiset implementation currently uses lowercase plus whitespace splitting
(`text.lower().split()`). This is a known normalization limitation. The config enforces a 100% match
under that implementation; no real Winoground data was available to evaluate the observed rate.

## 6. Baseline results

All values below are status markers, not numeric zeros and not literature values.

### Table 1: Standard retrieval

| Model | Benchmark | i2t R@1/R@5/R@10 | t2i R@1/R@5/R@10 | Mean recall | Status |
| --- | --- | --- | --- | --- | --- |
| OpenAI ViT-B/32 | Flickr30K | NOT RUN | NOT RUN | NOT RUN | blocked by missing data |
| OpenAI ViT-B/32 | MS COCO | NOT RUN | NOT RUN | NOT RUN | blocked by missing data |
| OpenCLIP ViT-B/32 LAION-2B | Flickr30K | NOT RUN | NOT RUN | NOT RUN | blocked by missing data |
| OpenCLIP ViT-B/32 LAION-2B | MS COCO | NOT RUN | NOT RUN | NOT RUN | blocked by missing data |
| OpenCLIP ViT-B/16 LAION-2B | Flickr30K | NOT RUN | NOT RUN | NOT RUN | blocked by missing data |
| OpenCLIP ViT-B/16 LAION-2B | MS COCO | NOT RUN | NOT RUN | NOT RUN | blocked by missing data |

### Table 2: Compositional evaluation

| Model | SugarCrepe overall/macro | ARO four subsets/macro | Winoground image/text/group | BiVLC image/text/group | Status |
| --- | --- | --- | --- | --- | --- |
| OpenAI ViT-B/32 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | blocked by missing data |
| OpenCLIP ViT-B/32 LAION-2B | NOT RUN | NOT RUN | NOT RUN | NOT RUN | blocked by missing data |
| OpenCLIP ViT-B/16 LAION-2B | NOT RUN | NOT RUN | NOT RUN | NOT RUN | blocked by missing data |

### Table 3: Reliability and efficiency

| Model | Benchmark set | Tie/margin/blind controls | Runtime | Peak VRAM | Cache | Embedding dimension | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OpenAI ViT-B/32 | all six | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | blocked by missing data |
| OpenCLIP ViT-B/32 LAION-2B | all six | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | blocked by missing data |
| OpenCLIP ViT-B/16 LAION-2B | all six | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | blocked by missing data |

No run is `complete`, `reportable`, `failed`, or `locally reproduced`; all 18 are `not run` and
`blocked by missing data`. The successful smoke test is an environment test, not a benchmark run.

## 7. Prediction audit

Prediction rows reviewed: 0. No `predictions.jsonl` exists because no real run was started. Indexing,
ties, margins, directionality failures, 2x2 diagonal conventions, possible annotation issues, and
human visual review samples therefore remain NOT RUN. The prediction review CSV intentionally has
only the agreed schema header; it contains no invented sample IDs or automatic failure labels.

## 8. Code changes

- `.gitignore`: ignore `data/aro/`, `data/winoground/`, and `data/bivlc/` so authorized data cannot be
  accidentally staged. Risk is low and limited to Git tracking behavior; metrics are unaffected.
- `tests/test_repository_hygiene.py`: parameterized regression test verifies all six real dataset
  roots are ignored using Git's own matcher. It requires the test checkout to be a Git worktree.
- `reports/phase1_baseline_audit.md`: auditable narrative and three required result tables.
- `reports/phase1_baseline_metrics.csv`: all 18 planned runs with explicit blocked/not-run status.
- `reports/phase1_baseline_failures.csv`: one exact missing-data blocker per benchmark.
- `reports/phase1_prediction_review.csv`: taxonomy-compatible header only; zero fabricated reviews.

No evaluator, metric, schema, benchmark annotation, model configuration, or previous result was
changed. No run needs rerunning because no real run was generated.

## 9. Remaining blockers

Prepare data only from authorized upstream copies. Replace the source and license descriptions below
with reviewed factual text before running each command.

```bash
recoalign prepare-flickr30k --karpathy-json data/flickr30k/incoming/dataset_flickr30k.json --dataset-root data/flickr30k --manifest-output manifests/datasets/flickr30k.yaml --source "REPLACE WITH VERIFIED SOURCE" --license "REPLACE WITH VERIFIED LICENSE NOTE" --hash-images
recoalign prepare-coco --karpathy-json data/mscoco/incoming/dataset_coco.json --dataset-root data/mscoco --manifest-output manifests/datasets/mscoco.yaml --source "REPLACE WITH VERIFIED SOURCE" --license "REPLACE WITH VERIFIED LICENSE NOTE" --hash-images
recoalign prepare-sugarcrepe --official-data-dir data/sugarcrepe/incoming --dataset-root data/sugarcrepe --manifest-output manifests/datasets/sugarcrepe.yaml --source "Official RAIVNLab SugarCrepe release, verified locally" --license "REPLACE WITH VERIFIED SUGARCREPE AND COCO LICENSE NOTE" --hash-images
recoalign prepare-aro --source-jsonl data/aro/incoming/aro.jsonl --dataset-root data/aro --manifest-output manifests/datasets/aro.yaml --source "REPLACE WITH VERIFIED OFFICIAL ARO EXPORT" --license "REPLACE WITH VERIFIED UPSTREAM TERMS" --hash-images
recoalign prepare-winoground --source-jsonl data/winoground/incoming/winoground.jsonl --dataset-root data/winoground --manifest-output manifests/datasets/winoground.yaml --source "REPLACE WITH VERIFIED OFFICIAL WINOGROUND EXPORT" --license "REPLACE WITH VERIFIED UPSTREAM TERMS" --hash-images
recoalign prepare-bivlc --source-jsonl data/bivlc/incoming/bivlc.jsonl --dataset-root data/bivlc --manifest-output manifests/datasets/bivlc.yaml --source "REPLACE WITH VERIFIED HUMAN-FILTERED BIVLC EXPORT" --license "REPLACE WITH VERIFIED UPSTREAM TERMS" --hash-images
```

After preparation, verify all six manifests before any baseline command:

```bash
for dataset in flickr30k mscoco sugarcrepe aro winoground bivlc; do
  recoalign verify-dataset --manifest "manifests/datasets/${dataset}.yaml" --root "data/${dataset}" || exit 1
done
```

Then run only the first model, one process at a time:

```bash
for benchmark in flickr30k mscoco sugarcrepe aro winoground bivlc; do
  recoalign run-baseline --config "configs/baseline/openclip_vit_b32_laion2b_${benchmark}.yaml" || exit 1
done
```

Do not begin no-cache reruns until each of those six run directories passes provenance, schema,
prediction-count, finite-metric, norm, timing, and GPU-memory inspection.

## 10. Recommended next action

**First fix the data export.** Acquire the six benchmarks through their authorized official channels,
place them in the documented incoming/image layouts, record truthful source and license notes, and run
the six preparation commands with `--hash-images`. Commit only the resulting lightweight reviewed
manifests when licensing permits; do not commit images or annotations. Once all six verification
commands pass, execute the OpenCLIP ViT-B/32 LAION-2B six-benchmark gate before considering the full
18-run matrix or starting failure taxonomy work.
