# Model Comparison Protocol

## Evaluation matrix

The committed matrix contains five backbones (including ReferenceVLM) by three unchanged
experiments. All cells share the same dataset generator, split manifest, prompts, answer evaluator,
decoding constraints, and metrics. Cross-model conclusions require the same completed seed set.

## Comparison rules

1. Never tune prompts per model or OOD split.
2. Never compare a dry-run, single-seed run, or unpinned adapter with claim-bearing evidence.
3. Report every experiment and failed split; do not select only favorable cells.
4. Compare Graph vs Caption and correct vs corrupted/random structure within the same model first.
5. Use cross-model consistency to assess generality, not to rank models by raw accuracy.
6. Keep ReferenceVLM visible as a pipeline sanity check, clearly labelled non-neural.

## Required provenance equality

Before aggregating models, verify experiment ID/version, dataset manifest hash, prompt protocol hash,
generation settings, seed set, and git state. Model-specific checkpoint, dtype, device mapping, and
chat wrapper are expected differences and must be disclosed.

## Advancement criterion

Proceed to Structured Interface Diagnosis only after at least one pinned real backbone completes all
three experiments and the registered real-model evidence is reviewed. Broad architecture claims
require replicated eligible results from multiple model families. Adapter readiness alone is not a
reason to begin method design.
