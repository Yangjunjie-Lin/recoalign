# ReCoAlign research pivot plan

## Current stage

`Research Pivot / Hypothesis Revision`

The former Structured Reasoning Interface Gap is falsified under the frozen LLaVA evidence. This
plan does not continue the old method roadmap and does not authorize paper writing.

## Phase P0 — Preserve the failed direction

1. Retain EXP001–EXP004 protocols, configs, predictions, metrics, decisions, and manifests.
2. Mark H001 and H004 `falsified`, H002 `supported`, and H003 `retired` because its evidence is
   inconclusive and its prerequisite graph advantage failed.
3. Keep `research/frozen_hypothesis.yaml` as the immutable record of the rejected claim.
4. Prevent generated paper-package assets from silently returning the old claim to `pending` or
   `testing`.

Exit criterion: claim lifecycle, evidence map, and narrative documents agree with the frozen
outcomes.

## Phase P1 — Construct validity before mechanism claims

1. Audit SAS by layer, label family, probe capacity, chance normalization, and scene-disjoint split.
2. Separate “not recovered by this probe” from “not represented by the model.”
3. Validate semantic and relation corruptions without VLM inference.
4. Confirm fact and token equivalence for fluent-sentence and canonical-triple serializers.

Exit criterion: the diagnostic instruments and factorial inputs pass leakage, balance, corruption,
and sensitivity checks.

## Phase P2 — Minimal factorial falsification

Preregister PIVOT_EXP_A from `research/next_experiment_plan.md`:

- semantic evidence: correct vs label-swapped;
- relational evidence: correct vs relation-flipped;
- serialization: fluent facts vs canonical triples;
- image and question held fixed.

Estimate semantic and relation correctness effects, their difference-in-differences, serialization
effects, and hop-depth moderation. Numeric gates must be selected by power analysis before real-model
outcomes are observed.

Exit criterion: PH001 receives GO, NO-GO, or INCONCLUSIVE without post-result protocol changes.

## Phase P3 — Discriminate surviving alternatives

If PH001 fails, use the frozen discrimination plan rather than broadening it post hoc:

- PIVOT_EXP_B: relation-selective grounding;
- PIVOT_EXP_C: primitive versus composition decodability;
- PIVOT_EXP_D: cross-modal reasoning equivalence.

If PH001 passes, replicate the complete pattern on a second pinned eligible backbone before making
a cross-model claim.

Exit criterion: one diagnosis has replicated claim-eligible evidence and its nearest alternatives
have failed their registered predictions.

## Phase P4 — Method decision, not method assumption

Only after Phase P3 may the project decide whether a learnable intervention is scientifically
motivated. Any method must target the diagnosed bottleneck and receive a new freeze, training
registry, parameter-matched controls, and graph-free inference boundary. Existing TRAIN001–003 toy
assets do not satisfy this gate.

## Capability-preservation controls

OpenCLIP/retrieval pipelines and external benchmarks remain frozen controls. They may measure
collateral capability changes after a future intervention, but they are not evidence for PH001.

## Decision policy

- GO now means GO to preregistration of PIVOT_EXP_A only.
- NO-GO remains in force for model development, paper writing, and submission.
- Failed and inconclusive evidence stays visible.
- Venue targets never determine hypothesis or threshold choices.
