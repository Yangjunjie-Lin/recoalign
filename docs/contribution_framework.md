# ReCoAlign contribution framework

The contribution is frozen as a mechanism-driven research program. The repository does
not claim that every VLM fails in the same way or that the final method is complete.

## Contribution 1 — Mechanism discovery

Identify and operationalize the Structured Reasoning Interface Gap: visual semantics may
be present while structured accessibility to the reasoning path is insufficient.

Evidence: EXP001 and EXP002, subject to their preregistered controls and decision gates.

## Contribution 2 — Mechanistic diagnosis

Separate three stages that are often conflated:

1. visual semantic availability;
2. structured representation accessibility;
3. reasoning execution.

Evidence: EXP004 SAS, StAS, RES, graph reconstruction/probe results, and oracle-graph
interventions. Missing hidden-state APIs remain missing evidence.

## Contribution 3 — Learnable interface

Introduce ReCoAlign as a learnable visual semantic → structured reasoning interface. The
structure supervision used during training is not an oracle graph supplied to the model
at inference. The implementation must therefore expose structure tokens/context through
the model interface and pass the registered ablations.

Evidence: TRAIN001–TRAIN003, structure-token probes, parameter-matched controls, and
inference interventions.

## Contribution 4 — Generalization evaluation

Test whether the learned interface improves compositional reasoning and composition-
disjoint OOD generalization across registered VLMs and tasks while preserving ordinary
vision-language capability.

Evidence: EXP003 and the frozen comprehensive evaluation matrix. Missing or blocked cells
remain visible and are not imputed.

## Evidence boundary

The contribution framework is a mapping from claims to required evidence, not a claim
that all evidence is already complete. The current scientific readiness decision remains
NO-GO until the real-VLM and claim-eligible method evidence is present.
