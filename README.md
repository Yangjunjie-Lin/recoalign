# ReCoAlign

**Region-aware Compositional Alignment for Fine-grained Image–Text Retrieval**

ReCoAlign is a research-oriented framework for studying fine-grained and compositional image–text retrieval with OpenCLIP. The project investigates whether structure-aware supervision can improve attribute, relation, role, count, and word-order understanding while preserving conventional retrieval quality.

> **Project status:** Phase 0 — reproducible research infrastructure. No proposed ReCoAlign method or paper result should be treated as completed yet.

## Research question

Modern vision–language models are effective at global semantic matching, but they can fail when captions differ only in attributes, relations, object roles, counts, or word order. ReCoAlign asks whether explicit structural supervision and local visual evidence can improve bidirectional compositional retrieval without sacrificing standard retrieval, hard-positive robustness, or zero-shot transfer.

## Planned research scope

- Typed compositional hard positives and hard negatives with auditable provenance.
- Region–concept and relation-aware alignment over local visual tokens.
- Capability-preserving objectives that retain the pretrained model's general retrieval behavior.
- Bidirectional evaluation on standard retrieval and dedicated compositionality benchmarks.
- Reproducible configurations, manifests, ablations, statistical analysis, and visualizations.

## Initial experimental scope

| Component | Initial choice |
| --- | --- |
| Main framework | OpenCLIP |
| Local development model | ViT-B/32 |
| Standard retrieval | Flickr30K, MS COCO |
| Compositional evaluation | SugarCrepe, ARO, Winoground, VL-CheckList, BiVLC |
| Metrics | Text→Image and Image→Text R@1/R@5/R@10, mean recall, benchmark accuracy |
| Local hardware target | NVIDIA RTX 3060 Laptop GPU, 6 GB VRAM |

LAVIS-based baselines may be added in a separate environment after the OpenCLIP evaluation path is stable.

## Repository layout

```text
recoalign/
├── .github/workflows/          # Continuous integration
├── configs/                    # Versioned experiment intent
├── data/                       # Dataset instructions; raw data is ignored
├── docs/                       # Architecture and reproducibility contracts
├── manifests/                  # Dataset, checkpoint, and sweep provenance
├── paper/                      # Manuscript-controlled assets
├── results/                    # Lightweight reviewed result summaries
├── schemas/                    # Machine-readable record contracts
├── scripts/                    # Thin executable utilities
├── src/recoalign/
│   ├── analysis/               # Tables, statistics, and failure analysis
│   ├── benchmarks/             # Unified benchmark adapter contract
│   ├── data/                   # Dataset-neutral records and manifests
│   ├── evaluation/             # Retrieval metrics and serializers
│   ├── experiments/            # Run lifecycle and provenance
│   ├── generation/             # Future hard-positive/negative generation
│   ├── methods/                # Future ReCoAlign method variants
│   ├── models/                 # Encoder interfaces and OpenCLIP integration
│   └── training/               # Future training and checkpointing
└── tests/                      # CPU-only infrastructure and metric tests
```

See [`docs/architecture.md`](docs/architecture.md) for dependency boundaries and the experiment lifecycle.

## Environment

The recommended development environment is Ubuntu under WSL2 with Python 3.10. Install the CUDA-enabled PyTorch build selected for the current machine before installing the optional OpenCLIP dependencies.

```bash
conda create -n recoalign python=3.10 -y
conda activate recoalign

# Example only: choose the current CUDA-enabled command from PyTorch for the machine.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install -e ".[openclip,dev]"
python scripts/check_environment.py
python scripts/smoke_test_openclip.py --model ViT-B-32 --pretrained laion2b_s34b_b79k
pytest
```

For WSL2, install the NVIDIA driver on Windows and verify GPU access with `nvidia-smi` inside WSL. Do not install a separate Linux display driver inside WSL2.

## Phase-0 workflow

Validate a committed experiment configuration:

```bash
recoalign validate-config configs/baseline/openclip_vit_b32.yaml
```

Create a self-contained run directory containing the resolved configuration, configuration hash, Git commit, package versions, CUDA details, and timestamps:

```bash
recoalign init-run \
  --config configs/baseline/openclip_vit_b32.yaml \
  --output-root outputs
```

After an evaluation command produces a JSON metrics object, finalize the run explicitly:

```bash
recoalign finalize-run outputs/<run-id> \
  --metrics outputs/<run-id>/metrics.pending.json \
  --status complete
```

Only independently reviewed `reportable` runs are eligible for manuscript tables:

```bash
recoalign build-table \
  --results-root results \
  --status reportable \
  --metrics i2t_R@1 t2i_R@1 mean_recall
```

Dataset snapshots are verified through committed manifests rather than inferred from local folder names:

```bash
recoalign verify-dataset \
  --manifest manifests/datasets/<dataset>.yaml \
  --root data/raw/<dataset>
```

## Six-gigabyte VRAM preset

Start with conservative settings:

```yaml
model: ViT-B-32
image_size: 224
train_batch_size: 4
eval_batch_size: 32
precision: amp
grad_accumulation_steps: 8
freeze_vision_encoder: true
```

If memory remains insufficient, reduce batch size before changing model or input resolution. Full ViT-L/14 training, large-scale ablations, and multi-seed experiments should be moved to a server.

## Reproducibility rules

1. Every reported number must reference a committed configuration and configuration hash.
2. Dataset snapshots and checkpoints require explicit manifests and provenance.
3. Raw datasets, pretrained weights, predictions, and embeddings are never committed.
4. Main trained results should be repeated across multiple seeds when computationally feasible.
5. `pilot`, `partial`, `failed`, `complete`, and `reportable` are distinct statuses.
6. Generated examples must retain source ID, transformation type, generator version, and audit status.
7. Favorable metrics alone are never sufficient to promote a run to `reportable`.

The full contract is documented in [`docs/reproducibility.md`](docs/reproducibility.md).

## Roadmap

- [x] Establish the initial package, OpenCLIP wrapper, and retrieval metrics.
- [x] Add Phase-0 configuration validation, manifests, environment capture, run records, schemas, and result-table generation.
- [ ] Reproduce zero-shot OpenCLIP retrieval on a small validation subset.
- [ ] Implement full Flickr30K retrieval evaluation.
- [ ] Add SugarCrepe, ARO, Winoground, and BiVLC adapters.
- [ ] Complete baseline failure taxonomy and bidirectional diagnostics.
- [ ] Select the smallest method hypothesis supported by the diagnostics.
- [ ] Run multi-backbone, multi-seed, transfer, robustness, and ablation experiments.
- [ ] Prepare manuscript figures, tables, supplementary material, and release artifacts.

## Citation

A citation entry will be added only after the method and manuscript are finalized. Do not cite the current scaffold as a completed method.

## License

No open-source license has been selected yet. Until a license is added, standard copyright rules apply.
