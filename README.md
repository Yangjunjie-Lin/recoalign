# ReCoAlign

**Region-aware Compositional Alignment for Fine-grained Image–Text Retrieval**

ReCoAlign is a research-oriented framework for studying fine-grained and compositional image–text retrieval with OpenCLIP. The project will investigate compositional hard negatives, region–phrase alignment, and grounding-aware consistency while keeping experiments reproducible across standard retrieval and compositionality benchmarks.

> **Project status:** early research scaffold. The current repository establishes the baseline infrastructure; proposed research modules and experimental results will be added incrementally and must not be treated as completed claims.

## Research question

Modern vision–language models are effective at global semantic matching, but they can fail when captions differ only in attributes, relations, object roles, counts, or word order. ReCoAlign asks whether explicit compositional negatives and local region–phrase evidence can improve fine-grained retrieval without sacrificing standard retrieval quality.

## Planned contributions

- A typed compositional hard-negative pipeline for object, attribute, relation, counting, and order perturbations.
- Region-aware phrase alignment built on local visual tokens or grounded candidate regions.
- A retrieval objective that combines global contrastive alignment with local compositional consistency.
- Evaluation on standard retrieval datasets and dedicated compositionality benchmarks.
- Reproducible configurations, ablations, result manifests, and qualitative visualizations.

## Initial experimental scope

| Component | Initial choice |
| --- | --- |
| Main framework | OpenCLIP |
| Local development model | ViT-B/32 |
| Standard retrieval | Flickr30K, MS COCO |
| Compositional evaluation | SugarCrepe, ARO, Winoground, VL-CheckList |
| Metrics | Text→Image and Image→Text R@1/R@5/R@10, mean recall, benchmark accuracy |
| Local hardware target | NVIDIA RTX 3060 Laptop GPU, 6 GB VRAM |

LAVIS-based baselines may be added in a separate environment after the OpenCLIP baseline is stable.

## Repository layout

```text
recoalign/
├── .github/workflows/        # Continuous integration
├── configs/baseline/         # Versioned experiment configurations
├── data/                     # Dataset instructions; raw data is not committed
├── docs/                     # Research and reproducibility notes
├── paper/                    # Manuscript assets and table/figure manifests
├── results/                  # Lightweight result summaries, not large checkpoints
├── scripts/                  # Executable environment and baseline utilities
├── src/recoalign/            # Python package
└── tests/                    # Unit tests independent of large models/data
```

## Environment

The recommended development environment is Ubuntu under WSL2 with Python 3.10. Install the CUDA-enabled PyTorch build that matches the current system from the official PyTorch selector before installing this package.

```bash
conda create -n recoalign python=3.10 -y
conda activate recoalign

# Install the appropriate CUDA-enabled PyTorch build first.
# Example only; select the command appropriate for the machine.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install -e ".[openclip,dev]"
python scripts/check_environment.py
python scripts/smoke_test_openclip.py --model ViT-B-32 --pretrained laion2b_s34b_b79k
pytest
```

For WSL2, install the NVIDIA driver on Windows and verify GPU access with `nvidia-smi` inside WSL. Do not install a separate Linux display driver inside WSL2.

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

If memory remains insufficient, reduce the batch size before changing the model or input resolution. Full ViT-L/14 training, large-scale ablations, and multi-seed experiments should be moved to a server.

## Reproducibility rules

1. Every reported number must reference a committed configuration and checkpoint identifier.
2. Raw datasets and pretrained weights are never committed to Git.
3. Dataset splits must be recorded explicitly rather than inferred from local folders.
4. Main results should be repeated across multiple seeds where computationally feasible.
5. Failed, partial, and pilot experiments must be distinguishable from reportable results.
6. Generated hard negatives must retain provenance, perturbation type, and filtering status.

## Roadmap

- [x] Establish repository structure and OpenCLIP environment checks.
- [ ] Reproduce zero-shot OpenCLIP retrieval on a small validation subset.
- [ ] Implement full Flickr30K retrieval evaluation.
- [ ] Add SugarCrepe, ARO, and Winoground evaluation adapters.
- [ ] Build and audit typed compositional hard negatives.
- [ ] Add lightweight hard-negative fine-tuning and reranking baselines.
- [ ] Implement region–phrase alignment.
- [ ] Run multi-model, multi-seed, and ablation experiments.
- [ ] Prepare manuscript figures, tables, and release documentation.

## Citation

A citation entry will be added after the method and manuscript are finalized. Please do not cite the current scaffold as a completed method.

## License

No open-source license has been selected yet. Until a license is added, standard copyright rules apply.
