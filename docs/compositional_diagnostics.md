# Phase 1.1 compositional diagnostics

Phase 1.1 extends the frozen baseline matrix from standard retrieval and SugarCrepe to ARO,
Winoground, and BiVLC. It deliberately adds dataset-bias controls before any ReCoAlign training is
started.

## Benchmarks

### ARO

ARO evaluates four subsets:

- `vg_attribution`: object–attribute binding;
- `vg_relation`: subject–relation–object understanding;
- `coco_order`: sensitivity to word order on COCO captions;
- `flickr30k_order`: sensitivity to word order on Flickr30K captions.

The normalized record format is one JSON object per line:

```json
{
  "sample_id": "vg_relation:0",
  "image": "VG_100K/1.jpg",
  "captions": ["the person rides the horse", "the horse rides the person"],
  "correct_index": 0,
  "subset": "vg_relation",
  "tags": ["role_swap"],
  "metadata": {}
}
```

Metrics include strict top-1 accuracy, macro subset accuracy, tie rate, mean correct-vs-best-negative
margin, and three blind controls: majority answer position, shortest-caption preference, and
longest-caption preference. These controls are dataset audits, not model scores.

### Winoground

Each example contains two images and two captions. Correct matches are the diagonal of the score
matrix `scores[image_index, caption_index]`. The normalized format is:

```json
{
  "sample_id": "0",
  "image_0": "0_0.png",
  "image_1": "0_1.png",
  "caption_0": "the mug is on the grass",
  "caption_1": "the grass is on the mug",
  "category": "winoground",
  "tags": ["unusual_image"],
  "metadata": {}
}
```

The runner reports image-to-text accuracy, text-to-image accuracy, group accuracy, tie rate,
directional margins, category scores, and tag-level group scores. The committed configs require all
400 official examples and a 100% caption content-check match rate. Winoground uses
`casefolded_alphanumeric_character_multiset_v1`, recorded in the dataset manifest and evaluation
metadata as the canonical alphanumeric-character-multiset method. The older
`caption_token_multiset_*` fields remain deprecated compatibility aliases.

This check case-folds each caption, removes non-alphanumeric characters, and compares character
frequencies. It does not modify either caption, perform linguistic tokenization, segment morphemes,
or validate word order: for example, `dog` and `god` pass. It therefore cannot by itself establish
the official semantic invariant. Dataset identity additionally depends on a fixed Hugging Face
revision, the source annotation SHA-256, and the complete manifest SHA-256.

### BiVLC

BiVLC uses the same two-by-two score matrix but includes a positive and synthetic-negative image as
well as their corresponding captions. The runner reports the same directional and group metrics so
that image-to-text and text-to-image weaknesses cannot be hidden by a one-way evaluation.

## Preparing local exports

ReCoAlign does not download or redistribute benchmark assets. Export authorized copies to the
path-based JSONL formats above, place the referenced files under `<dataset-root>/images`, and run:

```bash
python scripts/export_winoground.py \
  --revision "$WINOGROUND_HF_REVISION" \
  --output-root data/winoground
```

`WINOGROUND_HF_REVISION` must be the reviewed 40-character dataset commit SHA, not `main` or a
guessed latest revision. Capture the export summary's `exported_at` value for preparation.

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
  --source "official Hugging Face Winoground export" \
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

The preparation commands preserve the source annotation, write canonical `annotations/test.jsonl`,
generate a test-image inventory, and create a manifest with SHA-256 hashes. For Winoground,
`downloaded_at` must be the RFC 3339 UTC timestamp from the export summary (or another explicitly
recorded export timestamp); it is never inferred from a filesystem clock.

For Winoground, file hashes alone are insufficient for reportability. A reportable run requires a
manifest generated from an official pinned 40-character Hugging Face revision, consistent exporter
metadata, a recorded UTC acquisition/export time, and the canonical 400-sample split. Manifests
marked `requires_regeneration_from_pinned_revision`, `synthetic_or_unverified`, or
`template_not_generated` are rejected by `promote-run`.

## Running the matrix

Each of the three frozen encoders has ARO, Winoground, and BiVLC configs. For example:

```bash
recoalign run-baseline \
  --config configs/baseline/openclip_vit_b32_laion2b_aro.yaml

recoalign run-baseline \
  --config configs/baseline/openclip_vit_b32_laion2b_winoground.yaml

recoalign run-baseline \
  --config configs/baseline/openclip_vit_b32_laion2b_bivlc.yaml
```

A cache-free rerun and manual inspection of incorrect predictions are required before promotion to
`reportable`.

## Interpretation rules

1. Do not average standard retrieval and compositional benchmark scores into one headline number.
2. Do not claim compositional improvement when a blind heuristic is competitive with the model.
3. Report Winoground/BiVLC image-to-text, text-to-image, and group scores together.
4. Treat ties as incorrect and report tie rate.
5. Break ARO results down by all four subsets.
6. Preserve per-sample predictions so failure categories can be inspected before selecting a method.

## Research references

- ARO: *When and why vision-language models behave like bags-of-words, and what to do about it?*
- Winoground: *Probing Vision and Language Models for Visio-Linguistic Compositionality*
- BiVLC: *Extending Vision-Language Compositionality Evaluation with Text-to-Image Retrieval*
- Bias audit: *A Good CREPE needs more than just Sugar: Investigating Biases in Compositional
  Vision-Language Benchmarks*
