# ReCoAlign Phase 0-B — Minimal Semantic Utilization Validation

本目录实现一个冻结、可证伪的机制实验：`Za` 中已可线性解码的 object / attribute / relation / composition 语义，在进入 LLaVA-1.5-7B 的 Vicuna LLM 后，是否出现针对 relation / composition 的选择性下降。

这不是 benchmark，也不训练或优化模型。代码不会添加 adapter、loss 或修改 LLaVA 结构；唯一拟合的参数是离线诊断用 logistic-regression probe。

## Registered boundary

```text
image
  → frozen CLIP ViT-L/14-336 layer -2 patch tokens
  → frozen official LLaVA mlp2x_gelu projector (Za, 576 × 4096)
  → frozen Vicuna-7B decoder blocks
  → visual-token hidden means at states 8 / 16 / 24 / 32
  → decision-position hidden states and attention (supporting diagnostics)
```

Hidden-state index 0 is the LLM input embedding. At the 576 visual positions it must reproduce Za up to fp16 storage tolerance; this is an automatic integrity gate. State `k` is after `k` decoder blocks, with state 32 after final normalization.

## Data and probe

The run reuses the completed Phase 0-A synthetic dataset and selects a fixed balanced subset:

- 720 training images: 10 per each of 72 composition classes;
- 288 held-out test images: 4 per composition class;
- 24 test images for a separate, descriptive attention audit;
- fixed seed `20260818`;
- standardized multinomial logistic regression (`C=1`) at every boundary;
- 2,000 paired held-out bootstrap resamples.

The subset is intentionally the smallest registered design that gives every composition class repeated train and test observations. A positive result is only a controlled mechanism signal and requires higher-precision replication.

## Automatic GO / NO-GO

`GO` requires all protocol and Za-availability gates plus at least one of states 8/16/24/32 satisfying every condition at the same layer:

1. Za probe accuracy is at least 80% for all four semantics;
2. object drop is below 10 percentage points;
3. relation drop is above 15 percentage points and its paired 95% CI is above zero;
4. composition drop is above 15 percentage points and its paired 95% CI is above zero;
5. relation drop minus object drop has paired-bootstrap 95% CI above zero;
6. composition drop minus object drop has paired-bootstrap 95% CI above zero;
7. the full registered sample, layer, attention, bootstrap, and NF4 protocol ran;
8. all probes converged and the Za/state-0 identity check passed.

Every other result is `NO-GO`. Decision-position probes and attention are deliberately excluded from the GO rule because neither attention nor decodability alone demonstrates causal use.

## Precision constraint

The RTX 3060 Laptop GPU has 6GB VRAM, so the frozen 7B language weights use NF4 with contiguous CPU offload where necessary. NF4 changes numerical precision but does not update parameters. A `GO` would justify, not replace, a bf16/fp16 replication before causal intervention. A `NO-GO` applies to this registered controlled setup and must not be generalized to every VLM or natural-image distribution.

The registered contiguous device map keeps the embedding and first 28 NF4 decoder blocks on CUDA, then executes the final four frozen blocks, norm, and LM head in fp16 on CPU. This avoids a Windows Accelerate meta-tensor bug while preserving the exact layer order. Device placement is recorded in `llm_manifest.json`.

## Install and run

From the repository root on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r recoalign_phase0B_validation\requirements.txt
.\.venv\Scripts\python.exe recoalign_phase0B_validation\download_model.py
.\.venv\Scripts\python.exe recoalign_phase0B_validation\run.py
```

Projected 576-token tensors occupy about 4.8GB and are ignored by git. Extraction and LLM features are cached; the LLM loop records resumable progress every five samples.

## Outputs

```text
recoalign_phase0B_validation/
├── README.md
├── configs/
│   ├── protocol.json
│   └── selection_manifest.json
├── features/
│   ├── index.jsonl
│   ├── za_tokens.npy
│   ├── za_mean.npy
│   ├── visual_layer_*.npy
│   ├── decision_layer_*.npy
│   ├── attention.jsonl
│   └── *_manifest.json
├── results/
│   ├── results.csv
│   ├── statistics.json
│   ├── decision.json
│   └── predictions/
├── figures/
│   ├── semantic_transition_curve.png
│   ├── semantic_drop_heatmap.png
│   └── attention_analysis.png
└── ReCoAlign_Phase0B_Report.md
```

`decision.json` is authoritative; the Markdown report never overrides it post hoc.
