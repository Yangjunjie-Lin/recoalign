# ReCoAlign

**Region-aware Compositional Alignment for Fine-grained Image–Text Retrieval**

ReCoAlign is a research framework for testing whether structure-aware supervision can improve
attribute, relation, role, count, and word-order understanding while preserving conventional
image–text retrieval quality.

> **Project status:** Phase 1.1 diagnostic baseline implemented. The repository supports six
> benchmarks across three frozen OpenCLIP-compatible encoders. Real benchmark numbers still require
> authorized local data and GPU execution; no ReCoAlign method or paper result is complete yet.

## Research question

Modern vision–language models can match broad image semantics while failing when captions differ only
in attributes, relations, object roles, counts, or word order. ReCoAlign asks whether explicit
structural supervision and local visual evidence can improve bidirectional compositional retrieval
without sacrificing standard retrieval, hard-positive robustness, or zero-shot transfer.

## Frozen baseline matrix

Three encoders are evaluated without task-specific training:

| Model ID | Architecture | Pretrained weights |
| --- | --- | --- |
| `openai_vit_b32` | ViT-B/32 | original OpenAI CLIP |
| `openclip_vit_b32_laion2b` | ViT-B/32 | `laion2b_s34b_b79k` |
| `openclip_vit_b16_laion2b` | ViT-B/16 | `laion2b_s34b_b88k` |

Each model has committed configs for six benchmarks:

- **Flickr30K Karpathy 1K test:** bidirectional R@1/R@5/R@10, mean recall, median rank, mean rank;
- **MS COCO Karpathy 5K test:** the same retrieval protocol at larger scale;
- **SugarCrepe:** overall, macro-category, seven category accuracies, and tie rate;
- **ARO:** four subset accuracies, macro subset accuracy, margins, ties, and blind heuristics;
- **Winoground:** image-to-text, text-to-image, group, tag, margin, and tie metrics;
- **BiVLC:** the same bidirectional two-by-two protocol for positive/negative images and captions.

See [`docs/baseline_protocol.md`](docs/baseline_protocol.md) and
[`docs/compositional_diagnostics.md`](docs/compositional_diagnostics.md).

## Repository layout

```text
recoalign/
├── configs/baseline/           # 18 frozen model × benchmark configurations
├── data/                       # Local datasets and caches; ignored by Git
├── docs/                       # Protocol, architecture, and reproducibility contracts
├── environments/              # Versioned bootstrap profiles
├── manifests/                 # Dataset and checkpoint provenance
├── results/                   # Reviewed lightweight run summaries
├── schemas/                   # Runtime-enforced research record contracts
├── src/recoalign/
│   ├── benchmarks/             # Retrieval, pairwise, multi-choice, and 2×2 records
│   ├── data/                   # Manifest and dataset preparation
│   ├── evaluation/             # Cache, retrieval, compositional, and diagnostic metrics
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

Every run records package, PyTorch, CUDA, driver, GPU, config, Git, dataset manifest, and checkpoint
manifest identities.

## Prepare datasets

ReCoAlign does not redistribute images or benchmark annotations. Place authorized local copies in the
documented layout and create hashed manifests. Existing commands cover Flickr30K, MS COCO, and
SugarCrepe. Phase 1.1 adds:

```bash
recoalign prepare-aro \
  --source-jsonl data/aro/incoming/aro.jsonl \
  --dataset-root data/aro \
  --manifest-output manifests/datasets/aro.yaml \
  --source "official ARO export" \
  --license "upstream terms verified locally" \
  --hash-images

recoalign prepare-winoground \
  --source-jsonl data/winoground/incoming/winoground.jsonl \
  --dataset-root data/winoground \
  --manifest-output manifests/datasets/winoground.yaml \
  --source "official Winoground export" \
  --license "official gated research-use terms reviewed locally; upstream restrictions apply" \
  --source-revision "$WINOGROUND_HF_REVISION" \
  --exporter-version winoground-hf-export-v2 \
  --downloaded-at "$WINOGROUND_EXPORTED_AT" \
  --hash-images

recoalign prepare-bivlc \
  --source-jsonl data/bivlc/incoming/bivlc.jsonl \
  --dataset-root data/bivlc \
  --manifest-output manifests/datasets/bivlc.yaml \
  --source "official human-filtered BiVLC export" \
  --license "upstream terms verified locally" \
  --hash-images
```

Detailed formats are in [`docs/data_preparation.md`](docs/data_preparation.md) and
[`docs/compositional_diagnostics.md`](docs/compositional_diagnostics.md).

## Run a baseline

```bash
recoalign run-baseline \
  --config configs/baseline/openclip_vit_b32_laion2b_flickr30k.yaml

recoalign run-baseline \
  --config configs/baseline/openclip_vit_b32_laion2b_aro.yaml

recoalign run-baseline \
  --config configs/baseline/openclip_vit_b32_laion2b_winoground.yaml
```

The command creates a provenance-complete run, uses identity-bound embedding caches, writes aggregate
metrics and per-sample predictions, and finalizes the run as `complete`. Use `--no-cache` for an
independent verification rerun.

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

## Reportability

A technically complete run is not automatically a paper claim. Promotion requires a clean Git state,
valid manifests, a hash-verified test-image inventory, finite schema-valid metrics, a cache-free
verification rerun, prediction inspection, and reviewer identity. A Winoground run cannot become
reportable from reviewer notes alone: promotion recomputes the cached/no-cache comparison and the
prediction decisions and metrics, requires a full 400-sample mapping review, verifies exact
annotation-to-inventory coverage, and snapshots and hashes the resulting evidence.
Both the canonical and cache-disabled verification runs undergo the same config, environment,
manifest, annotation, prediction, decision, and metric integrity checks. Predictions must align
row-for-row with the normalized annotation, including sample ID, category, and tags.

A reportable Winoground result is revalidated when collected for tables. The collector verifies the
promotion evidence hashes, comparison gates, 400-row review evidence, canonical artifact digests,
and the retained complete verification run. A hand-edited `run.json` is not sufficient for inclusion.

```bash
recoalign promote-run outputs/<run-id> \
  --verification-run outputs/<no-cache-run-id> \
  --prediction-review reports/experiments/winoground/reviewed_sample_ids.csv \
  --reviewed-by "Yangjunjie Lin" \
  --notes "Checked indexing, manifests, blind controls, predictions, and cache-free rerun."
```

For Winoground, the verification run must remain `complete` and cache-disabled; only the canonical
cache-enabled run is promoted. A user-supplied comparison JSON is not accepted as evidence.

Interpretation rules:

1. Standard retrieval and compositional scores remain separate.
2. ARO is always reported by all four subsets and alongside blind heuristics.
3. Winoground and BiVLC report both directions and group accuracy together.
4. Ties are incorrect and tie rate is explicit.
5. Favorable metrics alone never make a run reportable.

## Roadmap

- [x] Complete Phase 0 provenance, schema, environment, and promotion infrastructure.
- [x] Build the 3-model × 3-benchmark Baseline v1 for Flickr30K, COCO, and SugarCrepe.
- [x] Add ARO, Winoground, and BiVLC with bias controls and failure-oriented predictions.
- [ ] Produce and independently review all 18 real baseline runs.
- [ ] Reproduce FLAIR and Concept-Centric CLIP through official inference implementations.
- [ ] Complete the failure taxonomy and select the smallest supported ReCoAlign hypothesis.
- [ ] Run multi-backbone, multi-seed, transfer, robustness, and ablation experiments.

## License

ReCoAlign source code is licensed under Apache-2.0. Third-party code, datasets, pretrained weights,
and generated assets remain subject to their own terms.
