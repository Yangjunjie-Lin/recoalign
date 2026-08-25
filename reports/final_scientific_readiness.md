# Final Scientific Readiness After PIVOT_EXP_A3P

## Program decision

**TERMINATE_CURRENT_PROGRAM**

The current ReCoAlign method and mechanism line is scientifically closed. It is not ready for
claim-bearing paper writing or submission.

## Frozen evidence integrity

- Frozen baseline HEAD: `7567d3a579384bcac76eb2b30dd6de538c62efe9`.
- A3P artifact manifest: 22/22 byte counts and SHA-256 digests independently valid.
- Primary predictions: 27,000/27,000.
- Complete seeds: 5/5, with 5,400 rows per seed.
- Figures: 8/8 readable.
- A3P decision and frozen result assets were not modified.

## Scientific lifecycle

| Item | Status |
| --- | --- |
| H001 Structured Reasoning Interface Gap | falsified and retired |
| H002 correct relational evidence sensitivity | supported_in_frozen_LLaVA_scope_only |
| H003 OOD Generalization | inconclusive and retired |
| H004 Interface Diagnosis | falsified and retired |
| PH001 joint semantic–structural causal explanation | unresolved_and_not_identified; retired |
| M1 semantic rescue | falsified and retired |
| M2 semantic rescue | falsified and retired |
| A3 full-field free generation | falsified as measurement instrument |
| A3R continuation free generation | falsified as measurement instrument |
| A3P forced-choice scorer/inventory | verified for measurement integrity only |

## Final pivot gate

No candidate passed all mandatory thresholds. PH005 had inadequate direct support and high post-hoc
risk; PH006 did not separate from established prompt sensitivity; PH007 could not exclude a floor
effect and overlapped mature modality-conflict work. No preregistration or power analysis was
created.

## Authorization

```yaml
selected_candidate: STOP
new_preregistration_allowed: false
real_inference_allowed: false
independent_backbone_replication_allowed: false
model_development_allowed: false
paper_writing_allowed: false
maximum_additional_diagnostic_pivots: 0
archive_and_release_allowed: true
```

The repository is ready only as a negative-results, reproducibility, governance, or internal
technical-report resource.
