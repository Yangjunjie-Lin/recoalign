# LLaVA Claim-Evidence Package

This directory publishes the first claim-eligible execution attempt for the frozen ReCoAlign
mechanism protocols on LLaVA-1.5-7B. It contains favorable, unfavorable, and rejected results.
Nothing here is evidence that a learned ReCoAlign interface works.

| Experiment | Execution | Registered decision | Raw predictions |
|---|---:|---:|---:|
| EXP001 Graph vs Text | complete, 5 seeds | NO-GO | 8,000 |
| EXP002 Structural Necessity | complete, 5 seeds | GO | 6,000 |
| EXP003 OOD Composition | integrity failure after seed 20260818 | INCONCLUSIVE | 1,920 unpromoted |
| EXP004 Interface Diagnosis | complete, 5 seeds | NO-GO | 360 |

Each completed experiment directory contains the authoritative `metrics.json`, registered
`decision_report.yaml`, figures, and gzip-compressed JSONL predictions. EXP003 retains the failed
bundle plus the complete first-seed metrics, split manifest, run record, and predictions, all marked
`unpromoted` because `random_graph_length_matched=false` failed the frozen integrity gate.

The compressed files can be inspected without changing their bytes:

```bash
gzip -dc reports/paper_evidence/EXP001/predictions.jsonl.gz | head
```

Artifact sizes, compressed and decompressed hashes, line counts, checkpoint identity, and runtime
identity are frozen in [`artifact_manifest.yaml`](artifact_manifest.yaml). Generated images,
checkpoint weights, inference caches, and local run directories remain ignored by Git.
Machine-local prefixes in the retained EXP003 run record are normalized to repository-relative
paths for publication; checkpoint, config, and prompt hashes remain unchanged.

The complete-run failure summaries are retained separately under
[`reports/failure_analysis/EXP001/llava_1_5_7b/EXP001/`](../failure_analysis/EXP001/llava_1_5_7b/EXP001/)
and [`reports/failure_analysis/EXP002/llava_1_5_7b/EXP002/`](../failure_analysis/EXP002/llava_1_5_7b/EXP002/).
Single-seed preflight/retry analyses and uncompressed error rows remain ignored process artifacts.

Scientific interpretation is deliberately conservative:

- EXP001 shows Caption outperforming Graph under both natural and token-matched controls.
- EXP002 shows strong sensitivity to missing, random, wrong, and opaque-label graph interventions.
- EXP003 cannot support an OOD claim because its length-control integrity assertion failed.
- EXP004 reports low semantic-probe availability and unavailable language-side visual hidden states;
  it therefore does not establish the proposed high-SAS/low-StAS interface pattern.

The combined paper-readiness decision is **NO-GO**.
