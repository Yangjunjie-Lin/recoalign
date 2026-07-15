# ReCoAlign

**Region-aware Compositional Alignment for Fine-grained Image–Text Retrieval**

ReCoAlign is a research-oriented framework for fine-grained and compositional image–text retrieval
with OpenCLIP. The project studies whether structure-aware supervision can improve attribute,
relation, role, count, and word-order understanding while preserving standard retrieval quality.

> **Project status:** Phase 0 complete — reproducible research infrastructure. No ReCoAlign method
> or paper result should be treated as completed yet. The next milestone is the Phase-1 OpenCLIP
> retrieval and compositional benchmark baseline.

## Research question

Modern vision–language models are effective at global semantic matching but can fail when captions
differ only in attributes, relations, object roles, counts, or word order. ReCoAlign asks whether
explicit structural supervision and local visual evidence can improve bidirectional compositional
retrieval without sacrificing standard retrieval, hard-positive robustness, or zero-shot transfer.

## Planned research scope

- Typed compositional hard positives and hard negatives with auditable provenance.
- Region–concept and relation-aware alignment over local visual tokens.
- Capability-preserving objectives that retain pretrained retrieval behavior.
- Bidirectional evaluation on standard retrieval and compositionality benchmarks.
- Reproducible configurations, manifests, ablations, statistics, and visualizations.

## Initial experimental scope

| Component | Initial choice |
| --- | --- |
| Main framework | OpenCLIP |
| Local development model | ViT-B/32 |
| Standard retrieval | Flickr30K, MS COCO |
| Compositional evaluation | SugarCrepe, ARO, Winoground, VL-CheckList, BiVLC |
| Metrics | Text→Image and Image→Text R@1/R@5/R@10, mean recall, benchmark accuracy |
| Local hardware target | NVIDIA RTX 3060 Laptop GPU, 6 GB VRAM |

LAVIS-based baselines may be added in a separate environment after the OpenCLIP evaluation path is
stable.

## Repository layout

```text
recoalign/
├── .github/workflows/          # CI and lifecycle validation
├── configs/                    # Versioned experiment intent
├── data/                       # Dataset instructions; raw data is ignored
├── docs/                       # Architecture and reproducibility contracts
├── environments/               # Versioned bootstrap profiles
├── manifests/                  # Dataset, checkpoint, and sweep provenance
├── paper/                      # Manuscript-controlled assets
├── results/                    # Lightweight reviewed summaries
├── schemas/                    # Runtime-enforced JSON contracts
├── scripts/                    # Thin executable utilities
├── src/recoalign/
│   ├── analysis/               # Tables, statistics, and failure analysis
│   ├── benchmarks/             # Unified benchmark adapter contract
│   ├── data/                   # Dataset-neutral records and manifests
│   ├── evaluation/             # Retrieval metrics and serializers
│   ├── experiments/            # Run lifecycle, promotion, and provenance
│   ├── generation/             # Future hard-positive/negative generation
│   ├── methods/                # Future ReCoAlign variants
│   ├── models/                 # Encoder interfaces and OpenCLIP integration
│   └── training/               # Future training and checkpointing
└── tests/                      # CPU-only infrastructure and metric tests
```

See [`docs/architecture.md`](docs/architecture.md) for dependency boundaries.

## Environment

The recommended local environment is Ubuntu under WSL2 with Python 3.10. A bootstrap profile is
provided in `environments/wsl2-cu126-py310.yml`.

```bash
conda env create -f environments/wsl2-cu126-py310.yml
conda activate recoalign
pip install -e .

python scripts/check_environment.py
python scripts/smoke_test_openclip.py \
  --model ViT-B-32 \
  --pretrained laion2b_s34b_b79k
pytest
```

Install the NVIDIA display driver on Windows and verify `nvidia-smi` inside WSL2. Do not install a
separate Linux display driver inside WSL2.

## Phase-0 workflow

Validate a committed configuration:

```bash
recoalign validate-config configs/baseline/openclip_vit_b32.yaml
```

Create a run. This validates and snapshots dataset/checkpoint manifests, verifies declared local
files, records manifest hashes, and captures Git/package/CUDA metadata:

```bash
recoalign init-run \
  --config configs/baseline/openclip_vit_b32.yaml \
  --output-root outputs
```

After evaluation produces a JSON metrics object, finalize the run. `finalize-run` cannot create a
reportable result:

```bash
recoalign finalize-run outputs/<run-id> \
  --metrics outputs/<run-id>/metrics.pending.json \
  --status complete
```

Promote a reviewed run only after the provenance gates pass:

```bash
recoalign promote-run outputs/<run-id> \
  --reviewed-by "Yangjunjie Lin" \
  --notes "Checked dataset split, checkpoint provenance, and metrics."
```

Build a manuscript table from reviewed runs:

```bash
recoalign build-table \
  --results-root results \
  --status reportable \
  --metrics i2t_R@1 t2i_R@1 mean_recall
```

Verify a dataset snapshot independently:

```bash
recoalign verify-dataset \
  --manifest manifests/datasets/<dataset>.yaml \
  --root data/raw/<dataset>
```

## Reportability gates

A run cannot be promoted when any of the following is true:

- Git commit is missing, inconsistent, dirty, or unknown;
- dataset files are undeclared, missing, or fail size/hash verification;
- checkpoint manifest verification fails;
- run, metrics, environment, or manifests violate their JSON Schemas;
- the run has not first reached `complete`;
- reviewer identity is absent.

The resolved configuration, manifest snapshots, environment record, run record, and metrics are
kept together in each run directory.

## Six-gigabyte VRAM preset

```yaml
model: ViT-B-32
image_size: 224
train_batch_size: 4
eval_batch_size: 32
precision: amp
grad_accumulation_steps: 8
freeze_vision_encoder: true
```

Reduce batch size before changing model or input resolution. Full ViT-L/14 training, multi-seed
ablations, and larger-scale experiments should move to a server.

## Reproducibility rules

1. Every reported number references a committed config and config hash.
2. Dataset and checkpoint manifests are validated, hashed, and snapshotted into each run.
3. Git dirty state and a digest of uncommitted changes are recorded.
4. Raw datasets, weights, predictions, and embeddings are never committed.
5. `pilot`, `partial`, `failed`, `complete`, and `reportable` remain distinct states.
6. Generated examples retain source, transformation, generator, filtering, and audit provenance.
7. Favorable metrics alone never qualify a run for manuscript use.

See [`docs/reproducibility.md`](docs/reproducibility.md) for the complete contract.

## Roadmap

- [x] Establish the package, OpenCLIP wrapper, and retrieval metrics.
- [x] Add validated configs, runtime schemas, manifests, environment capture, run records, review
      promotion, result-table generation, CI lifecycle tests, and Apache-2.0 licensing.
- [ ] Reproduce zero-shot OpenCLIP retrieval on a small validation subset.
- [ ] Implement full Flickr30K retrieval evaluation.
- [ ] Add SugarCrepe, ARO, Winoground, and BiVLC adapters.
- [ ] Complete baseline failure taxonomy and bidirectional diagnostics.
- [ ] Select the smallest method hypothesis supported by diagnostics.
- [ ] Run multi-backbone, multi-seed, transfer, robustness, and ablation experiments.
- [ ] Prepare manuscript figures, tables, supplementary material, and release artifacts.

## Citation

A citation entry will be added only after the method and manuscript are finalized. Do not cite the
current infrastructure as a completed research contribution.

## License

ReCoAlign source code is licensed under the Apache License 2.0. Third-party code, datasets,
pretrained weights, and generated assets remain subject to their own terms.
