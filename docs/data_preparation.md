# Baseline Data Preparation

Dataset downloads and licenses remain the user's responsibility. ReCoAlign normalizes authorized
local copies and creates manifests; it does not redistribute Flickr30K or COCO images.

## Flickr30K

Expected layout before preparation:

```text
data/flickr30k/
├── images/
│   └── *.jpg
└── incoming/
    └── dataset_flickr30k.json
```

Run:

```bash
recoalign prepare-flickr30k \
  --karpathy-json data/flickr30k/incoming/dataset_flickr30k.json \
  --dataset-root data/flickr30k \
  --manifest-output manifests/datasets/flickr30k.yaml \
  --source "Karpathy Flickr30K split from the authorized local source" \
  --license "Flickr30K terms verified locally" \
  --hash-images
```

The command writes normalized JSONL files under `annotations/`, preserves the source annotation under
`source/`, verifies every referenced image, and writes `inventories/images.jsonl`. The baseline
inventory covers the 1,000-image Karpathy test split.

## MS COCO

Expected layout before preparation:

```text
data/mscoco/
├── images/
│   ├── train2014/
│   └── val2014/
└── incoming/
    └── dataset_coco.json
```

Run:

```bash
recoalign prepare-coco \
  --karpathy-json data/mscoco/incoming/dataset_coco.json \
  --dataset-root data/mscoco \
  --manifest-output manifests/datasets/mscoco.yaml \
  --source "MS COCO Karpathy split from the authorized local source" \
  --license "MS COCO terms verified locally" \
  --hash-images
```

The baseline inventory covers the 5,000-image Karpathy test split. Normalized annotations preserve
the `train2014/` or `val2014/` relative image path from the source split.

## SugarCrepe

Expected layout before preparation:

```text
data/sugarcrepe/
├── images/
│   └── val2017/          # copied or symlinked COCO-2017 validation images
└── incoming/
    ├── add_att.json
    ├── add_obj.json
    ├── replace_att.json
    ├── replace_obj.json
    ├── replace_rel.json
    ├── swap_att.json
    └── swap_obj.json
```

Run:

```bash
recoalign prepare-sugarcrepe \
  --official-data-dir data/sugarcrepe/incoming \
  --dataset-root data/sugarcrepe \
  --manifest-output manifests/datasets/sugarcrepe.yaml \
  --source "Official RAIVNLab SugarCrepe release" \
  --license "SugarCrepe and COCO terms verified locally" \
  --hash-images
```

The normalized test file is written to `data/sugarcrepe/annotations/test.jsonl`.

## Verification

```bash
recoalign verify-dataset \
  --manifest manifests/datasets/flickr30k.yaml \
  --root data/flickr30k

recoalign verify-dataset \
  --manifest manifests/datasets/mscoco.yaml \
  --root data/mscoco

recoalign verify-dataset \
  --manifest manifests/datasets/sugarcrepe.yaml \
  --root data/sugarcrepe
```

Generated manifests are part of the experiment definition and should be reviewed and committed before
promoting results to `reportable`. A size-only inventory remains useful for pilot runs, but paper-ready
promotion requires `--hash-images`.
