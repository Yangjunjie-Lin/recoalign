# Frozen research hypothesis: rejected record

The machine-readable historical source is `research/frozen_hypothesis.yaml`.

## Frozen statement

> Vision-language models can preserve useful visual semantics while lacking a sufficiently
> accessible structured intermediate representation between visual perception and language
> reasoning; learning that interface can improve compositional reasoning and unseen-composition
> generalization.

## Status

**`falsified` on 2026-08-21 under the frozen LLaVA-1.5 evidence.**

- EXP001: NO-GO; graph underperforms controlled caption.
- EXP002: GO; correct supplied relations matter.
- EXP003: INCONCLUSIVE; integrity gate failed.
- EXP004: NO-GO; high-semantic/low-structured-access pattern absent.

EXP002 does not rescue the rejected statement because relation sensitivity does not establish graph
optimality, semantic preservation, or an internal interface gap.

## Successor candidate

The provisional successor is `semantic_structural_integration_failure_candidate_v1`, recorded in
`research/pivot_hypothesis.yaml`. It is not frozen as a supported hypothesis and does not authorize
paper writing or model development.

## Integrity rule

The rejected statement, its falsification gates, and all associated evidence remain in the
repository. Narrative migration may change active status but may not weaken, rewrite, or delete the
historical protocol.
