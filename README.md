# ReCoAlign

**Region-aware Compositional Alignment for Fine-grained Image–Text Retrieval**

ReCoAlign is a research framework for studying whether structure-aware supervision can improve
attribute, relation, role, count, and word-order understanding while preserving conventional
image–text retrieval quality.

> **Project status:** Phase 1 Baseline v1 implemented. The repository can prepare reproducible local
> Flickr30K, MS COCO, and SugarCrepe snapshots and evaluate a frozen OpenCLIP model matrix. No
> ReCoAlign method or paper result should yet be treated as completed.

## Research question

Modern vision–language models can match broad image semantics while failing when captions differ only
in attributes, relations, object roles, counts, or word order. ReCoAlign asks whether explicit
structural supervision and local visual evidence can improve bidirectional compositional retrieval
without sacrificing standard retrieval, hard-positive robustness, or zero-shot transfer.

## Baseline v1

The first trusted evidence table evaluates three frozen encoders:

| Model ID | Architecture | Pretrained weights |
| --- | --- | --- |
| `openai_vit_b32` | ViT-B/32 | original OpenAI CLIP |
| `openclip_vit_b32_laion2b` | ViT-B/32 | `laion2b_s34b_b79k` |
| `openclip_vit_b16_laion2b` | ViT-B/16 | `laion2b_s34b_b88k` |

Each model is evaluated on:

- **Flickr30K Karpathy 1K test:** image-to-text and text-to-image R@1/R@5/R@10, mean recall,
  median rank, and mean rank;
- **MS COCO Karpathy 5K test:** the same bidirectional retrieval metrics at larger scale;
- **SugarCrepe:** overall accuracy, macro-category accuracy, seven per-category accuracies, and tie
  rate.

See [`docs/baseline_protocol.md`](docs/baseline_protocol.md) for the exact protocol and the required
later comparators, including FLAIR, Concept-Centric CLIP, ARO, Winoground, and BiVLC.

## Repository layout

```text
recoalign/
├── configs/baseline/           # Nine frozen model × benchmark configurations
├── data/                       # Local datasets and caches; ignored by Git
├── docs/                       # Protocol, architecture, and reproducibility contracts
├── environments/              # Versioned bootstrap profiles
├── manifests/                 # Dataset and checkpoint provenance
├── results/                   # Reviewed lightweight run summaries
├── schemas/                   # Runtime-enforced research record contracts
├── src/recoalign/
│   ├── benchmarks/             # Normalized benchmark records
│   ├── data/                   # Manifest and dataset preparation
│   ├── evaluation/             # Cache, ranking, pairwise metrics, baseline runner
│   ├── experiments/            # Run lifecycle and reportability gates
│   └── models/                 # OpenCLIP inference adapter
└── tests/                      # CPU-only unit and synthetic end-to-end tests
```

## Environment

The local target is WSL2, Python 3.10, and an RTX 3060 Laptop GPU with 6 GB VRAM.

```bash
conda env create -f environments/wsl2-cu126-py310.yml
conda activate recoalign
pip install -e ".[openclip,dev]"

python scripts/check_environment.py
python scripts/smoke_test_openclip.py \
  --model ViT-B-32 \
  --pretrained laion2b_s34b_b79k
pytest
```

OpenCLIP is pinned for the baseline environment; every run additionally records the installed
package, PyTorch, CUDA, driver, GPU, config, Git commit, and manifest hashes.

## Prepare datasets

ReCoAlign does not redistribute dataset images. Place authorized local copies in the documented
layout, then normalize and manifest them:

```bash
recoalign prepare-flickr30k \
  --karpathy-json data/flickr30k/incoming/dataset_flickr30k.json \
  --dataset-root data/flickr30k \
  --manifest-output manifests/datasets/flickr30k.yaml \
  --source "authorized local Karpathy split" \
  --license "terms verified locally" \
  --hash-images

recoalign prepare-coco \
  --karpathy-json data/mscoco/incoming/dataset_coco.json \
  --dataset-root data/mscoco \
  --manifest-output manifests/datasets/mscoco.yaml \
  --source "authorized local MS COCO Karpathy split" \
  --license "MS COCO terms verified locally" \
  --hash-images

recoalign prepare-sugarcrepe \
  --official-data-dir data/sugarcrepe/incoming \
  --dataset-root data/sugarcrepe \
  --manifest-output manifests/datasets/sugarcrepe.yaml \
  --source "official RAIVNLab SugarCrepe release" \
  --license "SugarCrepe and COCO terms verified locally" \
  --hash-images
```

Detailed layouts are in [`docs/data_preparation.md`](docs/data_preparation.md).

## Run a baseline

One command creates a provenance-complete run, evaluates the model, writes aggregate metrics and
per-query predictions, and finalizes the run as `complete`:

```bash
recoalign run-baseline \
  --config configs/baseline/openclip_vit_b32_laion2b_flickr30k.yaml
```

Run SugarCrepe with the corresponding config:

```bash
recoalign run-baseline \
  --config configs/baseline/openclip_vit_b32_laion2b_sugarcrepe.yaml
```

Embedding caches are enabled by default and keyed by model identity, annotation digest, manifest
digest, split, and protocol version. Use `--no-cache` for a clean verification rerun.

A run directory contains:

```text
outputs/<run-id>/
├── config.resolved.yaml
├── environment.json
├── evaluation.json
├── metrics.json
├── predictions.jsonl
├── run.json
└── manifests/
    ├── checkpoint.yaml
    └── dataset.yaml
```

## Promote and tabulate

`run-baseline` produces a technically complete run, not a paper-ready claim. Promotion remains a
separate review action:

```bash
recoalign promote-run outputs/<run-id> \
  --reviewed-by "Yangjunjie Lin" \
  --notes "Checked indexing, manifests, cache-free rerun, predictions, and metrics."
```

Then copy reviewed lightweight run records under `results/` and generate tables:

```bash
recoalign build-table \
  --results-root results \
  --status reportable \
  --metrics i2t_R@1 i2t_R@5 i2t_R@10 t2i_R@1 t2i_R@5 t2i_R@10 mean_recall
```

## Reportability gates

A run cannot become `reportable` when Git is dirty or unknown, dataset inventory verification fails,
checkpoint provenance is invalid, runtime schemas fail, metrics are missing/non-finite, or reviewer
identity is absent. Favorable metrics alone never qualify a run for manuscript use.

## Six-gigabyte VRAM guidance

The supplied ViT-B/32 configs use image batch size 32 and text batch size 64. The ViT-B/16 configs use
image batch size 16. Reduce image batch size first if local inference exceeds available VRAM. Ranking
is performed in bounded CPU memory and does not require a full COCO-scale similarity matrix.

## Roadmap

- [x] Complete Phase 0 provenance, schema, environment, and promotion infrastructure.
- [x] Add normalized Flickr30K, MS COCO, and SugarCrepe preparation with image inventory verification.
- [x] Add OpenCLIP Baseline v1 with three checkpoints across nine runs, embedding caches, chunked
      retrieval ranks, pairwise compositional metrics, predictions, tests, and CI config validation.
- [ ] Produce and independently review all nine local baseline runs.
- [ ] Add ARO, Winoground, and BiVLC adapters and bias diagnostics.
- [ ] Reproduce FLAIR and Concept-Centric CLIP through their official inference implementations.
- [ ] Complete failure taxonomy and select the smallest supported ReCoAlign hypothesis.
- [ ] Run multi-backbone, multi-seed, transfer, robustness, and ablation experiments.

## License

ReCoAlign source code is licensed under Apache-2.0. Third-party code, datasets, pretrained weights,
and generated assets remain subject to their own terms.
