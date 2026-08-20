# Claim-Evidence Execution Report

## Outcome

The frozen LLaVA-1.5-7B EXP001–EXP004 execution is complete as an experimental decision cycle.
The result is not a scientific GO:

| Experiment | Execution status | Decision | Main observation |
|---|---|---|---|
| EXP001 | complete, 5 seeds, 8,000 predictions | NO-GO | Graph underperforms Caption |
| EXP002 | complete, 5 seeds, 6,000 predictions | GO | Correct graph structure is strongly necessary |
| EXP003 | first seed complete; bundle rejected | INCONCLUSIVE | Random-graph token-length integrity failed |
| EXP004 | complete, 5 seeds, 360 predictions | NO-GO | Low SAS; proposed interface pattern absent |

No prompt, split, seed, threshold, generation limit, or metric was changed after observing results.
Failed and unfavorable outcomes are retained.

## Execution identity

- Model: `liuhaotian/llava-v1.5-7b`
- Revision: `4481d270cc22fd5c4d1bb5df129622006ccd9234`
- Checkpoint fingerprint: `6334826897d616bad74eeec6a84cb43d62f5f7c3957628d73ddfaa5734e6c5c6`
- Loader / precision: `legacy_transformers`, NF4, FP16 compute
- Decoding: deterministic, temperature 0, one beam, 32 maximum new tokens
- Prompt hash: `1499f80e3c10d9cf2aaa2acdff143c0f77a9e76d18501cfa0a4e0b5aec4d7492`
- Source commit: `be69ca1fd2155a9339a0937ff327be9e4a5d3753`
- Runtime: Python 3.12.6, PyTorch 2.7.1+cu126, Transformers 4.52.4,
  bitsandbytes 0.48.2, CUDA 12.6
- Hardware: NVIDIA GeForce RTX 3060 Laptop GPU, 6,441,926,656 bytes

All nine checkpoint files matched the registered size and SHA-256 values before execution.

## EXP001: Graph vs Text

EXP001 completed all five preregistered seeds. Each seed contains 960 IID-primary predictions,
480 serialization/ordering ablation predictions, and 160 preliminary OOD predictions.

Main five-seed results:

| Condition | Natural accuracy | Token-matched accuracy |
|---|---:|---:|
| Image only | 0.3933 | 0.3933 |
| Object list | 0.3867 | 0.3867 |
| Caption | 0.9050 | 0.9117 |
| Scene graph | 0.8617 | 0.8333 |

Registered effects:

- natural Graph over Text: `-0.0433`, 95% CI `[-0.0583, -0.0283]`;
- token-matched Graph over Caption: `-0.0783`, 95% CI `[-0.1017, -0.0550]`;
- preliminary OOD token-matched Graph over Caption: `-0.1150`, 95% CI `[-0.1400, -0.0800]`;
- all five registered criteria failed in every seed.

Decision: **NO-GO**. This backbone does not show `Graph > Caption` under the frozen protocol.

## EXP002: Structural Necessity

EXP002 completed all five seeds and all 6,000 predictions. Full graph accuracy is `0.8533`.
Registered structure-dependency effects are:

| Contrast | Mean gain | 95% CI |
|---|---:|---:|
| Full over 25% removal | 0.2900 | [0.2767, 0.3033] |
| Full over 50% removal | 0.3017 | [0.2850, 0.3200] |
| Full over 75% removal | 0.3550 | [0.3367, 0.3767] |
| Full over random graph | 0.4517 | [0.4400, 0.4617] |
| Full over wrong graph | 0.1800 | [0.1583, 0.2033] |
| Full over randomized relation labels | 0.3567 | [0.3417, 0.3800] |

The depth-dependency slope is `0.1380`; all seven criteria passed with positive effects in 5/5
seeds. Decision: **GO**.

This supports sensitivity to supplied structure. It does not establish that the VLM internally
lacks a structured interface or that ReCoAlign learns one.

## EXP003: OOD Composition

The first registered seed generated all 1,920 predictions over IID, composition OOD, relation OOD,
and hop OOD. The split leakage and primitive-coverage assertions passed, but the preregistered
`random_graph_length_matched` assertion failed. The multi-seed bundle therefore stopped before
promotion and was recorded as failed/`INCONCLUSIVE`.

The unpromoted first-seed graph-over-caption effects were `-0.1667` on IID, `-0.0833` on
composition OOD, `0.0167` on relation OOD, and `-0.1333` on hop OOD. They are retained for diagnosis
only and are not paper-claim evidence.

No protocol repair was attempted because Prompt 14 froze token controls and forbade post-result
changes. Decision: **INCONCLUSIVE**.

## EXP004: Interface Diagnosis

EXP004 completed five seeds. The image-encoder and projected-token probes were available;
language-side visual hidden states and graph reconstruction were unavailable in the registered
adapter.

Five-seed scores:

- SAS: `0.1214`, 95% CI `[0.1071, 0.1357]`;
- StAS: `0.2931`, 95% CI `[0.2646, 0.3216]`;
- RES: `0.8500`, 95% CI `[0.8333, 0.8667]`;
- SAS − StAS interface gap: `-0.1717`, 95% CI `[-0.2073, -0.1360]`;
- oracle graph gain: `0.4417`, 95% CI `[0.4250, 0.4583]`.

The registered classifier returns `TYPE_A_SEMANTIC_FAILURE`, not a high-SAS/low-StAS Type B
interface failure. Decision: **NO-GO**.

## Integrity and publication

The tracked package under `reports/paper_evidence/` contains metrics, decisions, figures, and
compressed raw predictions. The compressed prediction files decompress to exactly 8,000, 6,000,
1,920, and 360 lines for EXP001–EXP004 respectively. Package and decompressed SHA-256 values are in
`artifact_manifest.yaml`.

Generated images, the 13.5 GB checkpoint, 12,000+ content-addressed cache entries, local preflights,
and complete runtime trees remain ignored by Git. This keeps process artifacts out of the release
while publishing the evidence needed to verify reported results.

## Scientific decision

The evidence cycle is **NO-GO** for the current Structured Reasoning Interface Gap claim and for
paper writing. EXP002 establishes graph-input sensitivity, but EXP001 does not establish graph
superiority, EXP004 selects semantic failure rather than interface failure, and EXP003 is blocked
by an integrity violation.
