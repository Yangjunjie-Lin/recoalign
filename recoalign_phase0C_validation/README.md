# ReCoAlign Phase 0-C — Minimal Semantic Access Validation

This directory contains the frozen, falsifiable test of a single mechanism claim:

> A VLM can make an incorrect semantic decision even though the task-relevant semantic variable remains linearly available at the same pre-answer decision position, and restoring a matched successful activation can causally recover that decision.

## Registered scope

- Frozen official `liuhaotian/llava-v1.5-7b` checkpoint; no fine-tuning, adapter, loss, or model-structure change.
- The immutable Phase 0-B selection: 720 balanced training scenes and 288 held-out test scenes.
- Primary semantic variable: the four spatial relations `left/right/above/below`.
- Availability and behavior use the same query and the same pre-answer decision position.
- Availability: a standardized linear probe trained only on the 720 training states.
- Behavior: teacher-force the tokenizer-identical shared answer-space token, then take the frozen LM head's argmax over the same four relation tokens at their actual answer position.
- Supporting attention analysis covers all 288 test scenes but is excluded from the decision rule.
- Causal activation patching uses a successful same-composition donor and a matched counterfactual control donor. The primary test is a full donor blend at layer 16.

The user-supplied task text omitted Sections 3–12. `configs/protocol.json` is therefore the reconstructed registration. Version 1 was written before any Phase 0-C result; versions 2 and 3 make the tokenizer-position and attention-backend integrity corrections documented below. Its SHA-256 is embedded in the authoritative decision artifact.

## Automatic GO / NO-GO

All five gates must pass:

1. The full registered run, all 288×32 attention reductions, all probes, and the intervention stage complete.
2. L32 held-out availability is at least 80%, with bootstrap lower bound above 25% chance.
3. L32 availability minus behavioral accuracy is at least 15 percentage points, with paired-bootstrap lower bound above zero.
4. There are at least 24 behavioral failures; L32 availability on those failures is at least 75%, with lower bound above chance.
5. On 24 eligible failures, the registered L16 α=1 positive patch recovers at least 15%, beats the matched control by at least 10 points, and both relevant lower bounds exceed zero.

Every other outcome is `NO-GO`. `results/decision.json` is authoritative.

### Assay integrity note

Protocol v1 incorrectly compared relation-token logits before Vicuna's shared answer-space token and was invalidated after tokenizer-position audit. Protocol v2 corrected the scoring/extraction position. Its audit then exposed one near-tie baseline flip because behavior-with-attention used eager attention while patching used SDPA. Protocol v3 isolates attention in a separate diagnostic forward and requires exact patch-baseline reproduction. All data, thresholds, registered layers, bootstrap count, donor matching, and substantive GO gates remain unchanged.

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe recoalign_phase0C_validation\run.py
```

The 4.8 GB projected-token tensor is reused without modification from Phase 0-B. Resumable hidden-state, behavior, attention, and intervention caches live under `features/` and are ignored by git.

## Outputs

```text
recoalign_phase0C_validation/
├── results/
│   ├── availability.csv
│   ├── behavior.csv
│   ├── attention_analysis.csv
│   ├── activation_patch.csv
│   ├── statistics.json
│   └── decision.json
├── figures/
│   ├── semantic_gap.png
│   ├── attention_difference.png
│   └── recovery_curve.png
├── Phase0C_Report.md
└── README.md
```
