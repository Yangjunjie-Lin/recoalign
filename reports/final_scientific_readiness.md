# Final Scientific Readiness Report

## Decision

**NO-GO for paper writing and scientific submission.**

The frozen real-model evidence cycle is complete enough to make a scientific decision without
changing the registered protocols. That decision is unfavorable: only EXP002 passes. EXP001 and
EXP004 falsify their registered claims, and EXP003 remains inconclusive after its preregistered
token-length integrity gate failed. No failed or unfavorable result was removed or repaired after
inspection.

## Frozen execution identity

- Model: `liuhaotian/llava-v1.5-7b`
- Revision: `4481d270cc22fd5c4d1bb5df129622006ccd9234`
- Loader / precision: `legacy_transformers`, NF4 quantization, FP16 compute
- Hardware: NVIDIA GeForce RTX 3060 Laptop GPU, 6,441,926,656 bytes VRAM
- Runtime: Python 3.12.6, PyTorch 2.7.1+cu126, Transformers 4.52.4, CUDA 12.6
- Execution commit: `be69ca1fd2155a9339a0937ff327be9e4a5d3753`
- Checkpoint fingerprint: `6334826897d616bad74eeec6a84cb43d62f5f7c3957628d73ddfaa5734e6c5c6`
- Checkpoint verification: 9/9 registered files matched both byte count and a freshly recomputed
  SHA-256 digest
- Evidence package integrity: PASS; four experiment records, artifact hashes, gzip hashes, raw
  line counts, and decompressed hashes validated

The explicit model record is `reports/paper_evidence/model_manifest.yaml`. Protocol, seed, model,
checkpoint, revision, and dataset identities are frozen in `experiments/frozen_registry.yaml`.

## Claim validation

| Claim | Evidence | Status |
| --- | --- | --- |
| C001: Graph improves reasoning over information-controlled text | EXP001, 5 seeds, 8,000 predictions. Natural Graph − Caption = −0.0433, 95% CI [−0.0583, −0.0283]; token-matched Graph − Caption = −0.0783, 95% CI [−0.1017, −0.0550]. | **FALSIFIED / NO-GO** |
| C002: Correct relational structure, rather than graph-like form, drives the supplied-graph effect | EXP002, 5 seeds, 6,000 predictions. Full − Random = +0.4517, 95% CI [+0.4400, +0.4617]; Full − Wrong = +0.1800, 95% CI [+0.1583, +0.2033]. All seven registered criteria pass in 5/5 seeds. | **VERIFIED for frozen LLaVA scope / GO** |
| C003: Graph advantage persists under composition, relation, and hop OOD | EXP003 produced 1,920 first-seed predictions, but `random_graph_length_matched=false` failed before multi-seed promotion. The retained first-seed values are diagnostic only. | **INCONCLUSIVE** |
| C004: Semantic availability is high while structured accessibility is limited | EXP004, 5 seeds, 360 predictions. SAS = 0.1214, StAS = 0.2931, RES = 0.8500, SAS − StAS = −0.1717. The registered classifier selects `TYPE_A_SEMANTIC_FAILURE`, not an interface failure. | **FALSIFIED / NO-GO** |
| C005: ReCoAlign learns a graph-free structured interface | No claim-eligible trained real-VLM ReCoAlign checkpoint or method comparison exists. | **INFRASTRUCTURE ONLY** |
| C006: ReCoAlign improves compositional reasoning across models/tasks without severe capability loss | Comprehensive matrix remains 0/432 completed cells. | **PENDING** |
| C007: Gains are causally attributable to the learned interface | Only toy evidence exists; no paired real-model ReCoAlign intervention is frozen. | **PENDING** |

## Experiment decisions

| Experiment | Execution | Registered decision | Claim eligibility |
| --- | --- | --- | --- |
| EXP001 | Complete, 5/5 seeds | NO-GO | Eligible falsifying evidence |
| EXP002 | Complete, 5/5 seeds | GO | Eligible supporting evidence within LLaVA-1.5 scope |
| EXP003 | Stopped after first complete seed | INCONCLUSIVE | Not promoted because the frozen integrity gate failed |
| EXP004 | Complete, 5/5 seeds | NO-GO | Eligible falsifying evidence |

## Package validation

`build-paper-package` and `validate-paper-package` both complete successfully at the engineering
artifact level. The validator independently reports:

- claim-evidence integrity: PASS, with no hash or line-count failures;
- engineering artifacts: ready;
- scientific decision: NO-GO;
- frozen real-VLM evidence: completed, decision NO-GO;
- evidence map: 1 verified, 2 falsified, 1 inconclusive, 1 infrastructure-only, 2 pending.

## Scientific interpretation

The experiment supports a narrow conclusion: LLaVA-1.5 uses correct externally supplied relational
structure and degrades predictably under graph corruption. It does not support the stronger claim
that graph serialization outperforms equivalent captions, nor the proposed high-semantic / low-
structured-access interface-gap mechanism. It provides no evidence that a trained ReCoAlign method
solves the problem.

Therefore the repository has reached **Real Evidence Frozen — Scientific NO-GO**, not
**Claim-Eligible Evidence Ready** for the planned paper. Prompt 15 must remain blocked.
