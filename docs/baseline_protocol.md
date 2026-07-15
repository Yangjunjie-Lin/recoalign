# Credible Baseline Protocol v1

This protocol defines the first locally reproduced evidence table for ReCoAlign. It is deliberately
limited to frozen zero-shot encoders: no prompt tuning, fine-tuning, reranking, or external detector is
allowed in a baseline run.

## Evidence basis

The protocol follows the official OpenCLIP inference pattern: switch the model to evaluation mode,
encode images and text separately, L2-normalize both modalities, and compare them with dot-product
similarity. The first compositional benchmark is SugarCrepe because it was designed to reduce
language-only artifacts in earlier hard-negative benchmarks. Standard retrieval remains necessary
because compositional gains that damage ordinary retrieval are not acceptable.

Relevant primary sources:

- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [SugarCrepe, NeurIPS 2023](https://github.com/RAIVNLab/sugar-crepe)
- [ARO](https://arxiv.org/abs/2210.01936)
- [Winoground](https://arxiv.org/abs/2204.03162)
- [BiVLC](https://arxiv.org/abs/2406.09952)
- [FLAIR, CVPR 2025](https://github.com/ExplainableML/flair)
- [Concept-Centric CLIP, 2026](https://github.com/SamsungLabs/concept_centric_clip)

## Locally reproduced model matrix

| ID | Model | Pretrained weights | Purpose |
| --- | --- | --- | --- |
| `openai_vit_b32` | ViT-B/32 | `openai` | canonical original CLIP reference |
| `openclip_vit_b32_laion2b` | ViT-B/32 | `laion2b_s34b_b79k` | same backbone, open-data scaling comparison |
| `openclip_vit_b16_laion2b` | ViT-B/16 | `laion2b_s34b_b88k` | stronger local backbone comparison |

All three models are evaluated with the exact same normalized annotations, image files, metrics,
cache implementation, and prediction serializer.

## Benchmark matrix

### Flickr30K standard retrieval

- Split: Karpathy 1K test split.
- Unit: one image with all associated captions as positives.
- Directions: image-to-text and text-to-image.
- Primary metrics: R@1, R@5, R@10, and mean recall across six recall values.
- Diagnostics: median rank and mean rank in both directions.
- Ranking: bounded-memory blocks; a full image-caption similarity matrix is not required.

### MS COCO standard retrieval

- Split: full Karpathy 5K test split, not a cherry-picked 1K fold.
- Unit and directions: identical to Flickr30K.
- Primary metrics: R@1, R@5, R@10, and six-value mean recall.
- Diagnostics: median rank, mean rank, wall-clock stages, embedding dimension, and norm error.
- Scale gate: exactly 5,000 images and 25,000 captions.

### SugarCrepe compositional retrieval

- Source: all seven official categories.
- Images: COCO-2017 validation images used by the official benchmark.
- Decision: the image score for the positive caption must be strictly greater than the hard negative.
- Primary metrics: weighted overall accuracy and unweighted macro-category accuracy.
- Diagnostics: per-category accuracy and exact-score tie rate.

The seven categories are `add_att`, `add_obj`, `replace_att`, `replace_obj`, `replace_rel`,
`swap_att`, and `swap_obj`.

## Reproducibility requirements

A number is not reportable unless the run contains:

1. a committed experiment config and config digest;
2. committed dataset and checkpoint manifests;
3. normalized annotation and test-image inventory SHA-256 hashes;
4. Git commit and clean working-tree state;
5. OpenCLIP, PyTorch, CUDA, driver, and GPU versions;
6. aggregate metrics plus per-query or per-sample predictions;
7. an independent `promote-run` review.

Embedding caches are keyed by model identity, dataset manifest, annotation digest, split, and
protocol version. Cache hits and misses are recorded in `evaluation.json`.

## External strong baselines

FLAIR and Concept-Centric CLIP are required later as strong method-level comparators. Their official
inference paths differ from plain OpenCLIP global matching, so they must not be silently loaded as if
they were ordinary OpenCLIP checkpoints. Their results should be labeled either:

- `locally_reproduced_official_code`, or
- `paper_reported`.

ARO, Winoground, and BiVLC are Phase-1.1 benchmark extensions. The common record and score APIs in
this repository are intended to support them without changing the provenance contract.

## Baseline acceptance gate

Baseline v1 is accepted only when all nine model-benchmark configurations complete, prediction files
are retained, cache-free reruns match cached runs within floating-point tolerance, and manually
inspected examples confirm image/caption indexing is correct.
