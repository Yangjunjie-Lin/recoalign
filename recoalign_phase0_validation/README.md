# ReCoAlign Phase 0 Minimal Validation

This repository runs one deliberately small mechanism experiment:

> Is semantic information that is linearly decodable from a frozen vision representation (`Zv`)
> selectively lost at the frozen cross-modal alignment representation (`Za`), rather than all
> semantic factors suffering a similar generic bottleneck?

It is not a benchmark, a VLM training recipe, or evidence of state-of-the-art performance. The run
ends in an automatic, conservative **GO** or **NO-GO** decision.

## Model path

The machine has an RTX 3060 Laptop GPU with 6 GB VRAM, although the Python environment inspected
while preparing this repository had a CPU-only PyTorch wheel. The default `auto` backend therefore
selects based on the active PyTorch runtime:

- CUDA available: the real LLaVA-1.5-7B vision tower
  (`openai/clip-vit-large-patch14-336`) plus the official 7B `mm_projector.bin` from
  `liuhaotian/llava-v1.5-7b`;
- CPU-only PyTorch: the permitted `openai/clip-vit-base-patch32` fallback and its trained
  `visual_projection`.

The preferred CUDA path preserves the required boundary:

```text
image → frozen LLaVA CLIP ViT-L patch tokens from layer −2 (Zv)
      → frozen official LLaVA mlp2x_gelu projector output (Za)
```

The CPU fallback preserves an equivalent boundary:

```text
image → frozen CLIP vision_model pooled output (Zv)
      → frozen, pretrained CLIP visual_projection (Za)
```

Both tensors are captured with PyTorch forward hooks. Neither projection is hand-designed or fitted
for this experiment, so the experiment cannot create a positive result by deliberately deleting
relation dimensions. The full 7B language model is not loaded because it is unnecessary for the
registered `Zv → Za` question and is not safely batchable in 6 GB VRAM; `Zl` is therefore recorded as
unavailable. No model parameter is updated.

## Dataset

The default run creates exactly 10,000 224×224 PNG images with Pillow:

- 8,000 stratified training images;
- 2,000 stratified held-out test images;
- 500 additional semantic-isolation pairs (1,000 images) used only for control evaluation.

Scenes contain two differently shaped, differently colored objects and one of four relations. The
main factorial has 72 balanced composition classes:

- object role-pairs: `circle|square`, `square|triangle`, `triangle|circle`;
- ordered attribute role-pairs: all six distinct pairs of red, blue, and green;
- relation: left, right, above, below;
- composition: the full object + attribute + relation sentence.

Using only one direction for each shape pair prevents identical pixels from receiving contradictory
relation labels (for example, “circle left of square” and “square right of circle”). Metadata is in
`synthetic_dataset/metadata.jsonl` and includes the required `object1`, `object2`, `attribute1`,
`attribute2`, `relation`, and `composition` fields plus explicit probe labels.

For every isolation pair, A and B have identical object identities, colors, sizes, background, and
rendering jitter. Only `left ↔ right` or `above ↔ below` changes.

## Install and run

Python 3.10+ is required. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py
```

To use the RTX 3060 on Windows, install a CUDA-enabled PyTorch wheel before the requirements file
(choose the currently supported wheel from the official PyTorch selector; CUDA 12.6 is compatible
with the detected 566.07 driver):

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python run.py                         # auto now selects LLaVA vision + projector
```

The first preferred run downloads the public LLaVA projector (about 42 MB) and its CLIP ViT-L/14-336
vision tower (subject to upstream terms). Feature extraction can take hours on CPU. Intermediate data
and features are cached; rerunning resumes from valid caches. Use `--force-data` or
`--force-features` only when an intentional rebuild is needed.

Useful options:

```bash
python run.py --backend llava --device cuda --batch-size 16
python run.py --backend clip --device cuda --batch-size 128
python run.py --smoke-test          # engineering check only; always ineligible for GO
./run.sh --device cuda
```

The no-argument command is the registered scientific protocol. Reducing its sample counts is not
exposed through the command line; `--smoke-test` writes to `_smoke_workspace/` and is prominently
marked underpowered.

## Probes and statistics

Separate scikit-learn logistic-regression probes predict Object, Attribute, Relation, and
Composition from `Zv` and `Za`. Each probe uses only standardized frozen features, fixed `C=1`, and
the fixed global seed `20260818`. The probe is linear and cannot modify either representation.

For each accuracy and semantic preservation drop,

```text
SPD(S) = Accuracy(Zv, S) − Accuracy(Za, S),
```

the run saves the empirical mean, bootstrap standard deviation, and percentile 95% confidence
interval from 2,000 paired resamples. A paired one-sided bootstrap test evaluates whether Relation
drop is greater than Object drop. Predictions are saved for sample-level auditing.

## Pre-registered decision

The automatic decision is **GO** only when every condition passes:

1. Object drop < 10 percentage points;
2. Attribute drop < 15 percentage points;
3. Relation drop > 25 percentage points;
4. Composition drop > 25 percentage points;
5. Relation drop is greater than Object drop with one-sided bootstrap `p < 0.05`;
6. the relation-isolation drop is positive and its 95% CI excludes zero;
7. the full 8k/2k + 500-pair + 2,000-bootstrap protocol was run.

All other outcomes are **NO-GO**. Similar losses across all factors are reported as a generic
bottleneck (Pattern B); no material loss is reported as no failure at this stage (Pattern C); other
outcomes are explicitly inconclusive. There is no post-hoc override.

## Outputs

After a successful run:

```text
experiment_results/
├── README.md
├── ReCoAlign_Phase0_Report.md
├── decision.json
├── results.csv
├── run_config.json
├── statistics.json
├── predictions/
└── figures/
    ├── figure1_semantic_preservation_curve.png
    ├── figure2_semantic_degradation_heatmap.png
    └── figure3_semantic_factor_comparison.png

features/
├── main/{zv,za}/features.npy
├── main/index.jsonl
├── control/{zv,za}/features.npy
└── control/index.jsonl
```

Each feature-index row contains `image_id`, row-addressable `Zv` and `Za`, optional `Zl` (null for
CLIP), and the complete `semantic_label`. Manifests bind feature caches to exact metadata hashes,
model name, dimensions, library versions, and checkpoint revision when exposed by Transformers.

## Interpretation boundary

A GO says the controlled CLIP projection exhibits selective linear decodability loss and justifies
a bounded LLaVA replication. It does not establish prevalence on natural images. A NO-GO means this
specific stage and synthetic setup did not provide the required evidence; the report will not claim
that Semantic Preservation Failure exists.
