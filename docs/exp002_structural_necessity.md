# EXP002: Graph Completeness and Structural Necessity

## Motivation

EXP001 asks whether an information-controlled graph beats text. EXP002 asks the stronger causal
question: does performance depend on the relations themselves? A graph-looking prompt that works
equally well after its relations are removed or made wrong would falsify the proposed mechanism.

## Hypothesis

H002 predicts orderly degradation under relation removal and semantic corruption, with increasing
Structural Dependency Score (SDS) at greater reasoning depth. SDS is always reported as Full minus
the named control on the same samples.

## Pre-implementation audit

The audit found that EXP001 and Synthetic Compositional World v2 already supply deterministic
world-first generation, complete oracle graphs, 1–4 hop metadata, exact query-supporting edge
indices, relation labels, fixed images/questions, frozen model adapters, governed multi-seed
decisions, and retained predictions. The old EXP002 path was only a generic generator-v1 wrapper:
it exposed one unspecified partial graph and one single-edge random corruption, used a paired
t-test, and did not produce corruption manifests, length controls, depth/relation analyses, figures,
or the required top-level bundle. EXP002 now binds to generator-v2 and its authoritative scene graph.

## Corruption design

`synthetic_world/corruption/` exports deterministic `remove_relation()`, `flip_relation()`,
`swap_entity()`, and `randomize_graph()` operations. Relation removal prioritizes the supporting
edges declared by question generation and realizes 25/50/75% using a recorded ceiling rule. Flips
use declared semantic opposites; swaps change endpoints without changing the node inventory; random
graphs sample valid nonidentical triples while preserving node and edge counts.

Every intervention writes both graphs, the operation, ratio, seed, selected indices, SHA-256 hashes,
and invariant flags. The corruption manifest is evaluation metadata only and is never injected into
the model prompt.

## Evaluation

The runner executes Image Only, Full, all Partial ratios, Relation Flip, Object Swap, Random Graph,
serialization-order randomization, and opaque-label randomization on every scene. Full and Random
are exactly token-matched with the active model tokenizer. Results include mean/std/95% CI, paired
bootstrap tests, hop slices, depth slopes, relation tables, and three publication-style figures.

The required bundle is:

```text
outputs/EXP002/
├── config.resolved.yaml
├── metrics.json
├── predictions.jsonl
├── corruption_manifest.json
├── run.json
├── decision_report.yaml
├── relation_breakdown.csv
├── figure1_accuracy_degradation.png
├── figure2_corruption_ratio.png
├── figure3_reasoning_depth_degradation.png
└── seeds/<seed>/...
```

## Decision criteria

The decision is conjunctive: Full must beat every Partial ratio, Random, Wrong Graph, and opaque
labels; structural dependency must grow with depth; results must be significant and stable across
five seeds; and corruption, leakage, and exact length controls must pass. ReferenceVLM is an
infrastructure fixture and therefore always yields an INCONCLUSIVE scientific decision. Only a
preregistered, pinned, frozen target VLM can produce GO or NO-GO evidence.
