# ReCoAlign research plan

The active research identity is frozen in [`research_identity.md`](research_identity.md) and
[`../research/frozen_hypothesis.yaml`](../research/frozen_hypothesis.yaml). This plan follows
the questions and falsification gates in that freeze; it is not a retrieval-method roadmap.

## Problem statement

Vision-language models may preserve object, attribute, and relation semantics while failing to
make those semantics accessible as a structured intermediate representation for language-side
compositional reasoning. ReCoAlign studies this Structured Reasoning Interface Gap and tests
whether a learned visual-to-structure interface improves reasoning and unseen-composition
generalization.

## Phase 1 — Controlled mechanism validation

1. Freeze the synthetic world, factor splits, information-equivalence rules, and leakage checks.
2. Run EXP001 to compare graph and information-matched text conditions.
3. Run EXP002 to test structural necessity with partial, corrupted, randomized, and length-matched
   controls.
4. Run EXP003 on composition-disjoint OOD splits with reproducible manifests.
5. Keep all decisions paired, seeded, and eligibility-aware; dry-runs and ReferenceVLM outputs
   remain infrastructure-only.

Exit criterion: the registered structure effect survives controls and OOD evaluation, or the
corresponding hypothesis is falsified and retained as a negative result.

## Phase 2 — Real-VLM diagnosis

1. Evaluate frozen VLM backbones through the unified `BaseVLM` interface.
2. Run EXP004 to measure semantic availability (SAS), structured accessibility (StAS), and
   reasoning execution (RES).
3. Separate image-only, oracle-structure, corrupted-structure, graph reconstruction, and latent
   probe results.
4. Record missing APIs, checkpoints, and blocked cells rather than substituting a reference model.

Exit criterion: any Interface Gap claim requires high semantic availability, lower structured
accessibility, positive oracle-structure gain, and a registered cross-model pattern.

## Phase 3 — Learnable ReCoAlign interface

1. Train visual-to-structure tokens with graph labels as training signal only.
2. Align learned structure context with the reasoning path while keeping baseline backbones,
   datasets, and decoding settings fair.
3. Evaluate structure removal, random tokens, parameter-matched controls, supervision ablations,
   and inference interventions.
4. Never provide an oracle graph as an inference-time replacement for the learned interface.

Exit criterion: improvement is attributable to learned structured access and survives OOD and
capability-preservation controls.

## Capability-preservation controls

The existing OpenCLIP/retrieval pipeline, Flickr30K/MS COCO/SugarCrepe/ARO/Winoground adapters,
and provenance gates remain frozen controls. They answer whether a reasoning intervention damages
ordinary vision-language capability; they are not evidence by themselves for the Structured
Reasoning Interface Gap.

## Scientific integrity

The target venue is not an evidence criterion. Failed, blocked, and pending experiments remain
visible, and the current submission decision stays `NO-GO` until claim-eligible real-VLM and method
evidence is present.
