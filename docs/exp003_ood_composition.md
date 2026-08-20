# EXP003 OOD Compositional Generalization

## Motivation

EXP001 asks whether graphs are a better reasoning interface than lossless text. EXP002 asks whether
the model depends on the relations rather than graph-shaped formatting. EXP003 closes the remaining
memorization loophole: the relevant structure must transfer to combinations absent from the seen
partition.

## Split design

EXP003 writes `dataset/splits/{iid,composition_ood,relation_ood,hop_ood}`. Each family contains
`train/` and `test/` bundles plus a manifest under `dataset/splits/manifests/`.

| Split | Seen partition | Held-out partition | Controlled invariant |
|---|---|---|---|
| IID | known endpoint/relation/depth composition | same support, changed nuisance factors | composition is seen |
| Composition OOD | identity binding offset 0 | binding offsets 1/2 | every atomic factor is seen |
| Relation OOD | registered two-relation programs | disjoint two-relation programs | each relation and hop depth is seen |
| Hop OOD | 1–2 hops | 3–4 hops | identities and relation vocabulary are seen |

Mixed relation programs are queried as ordered edge sequences. They are not treated as a transitive
single relation, which would be semantically invalid.

## Leakage prevention

Allocation is constructive and seed-deterministic. Before evaluation the validator recomputes
answers from supporting graph edges, checks caption/graph fact-hash equivalence, verifies disjoint
sample IDs and held-out signatures, confirms test primitives are a subset of training primitives,
and validates every materialized file hash. Test outcomes are never read by the split builder.

## Evaluation

The frozen backend receives Image Only, Image + Caption, Image + Graph, or Image + Random Graph. The
correct and random graph prompts are required to have identical token counts. Predictions retain the
split, partition, template-pair ID, relation program, depth, and anti-memorization tag.

Reported outputs include accuracy, paired graph-caption and graph-random effects, IID–OOD gap,
retention ratio, depth curve, seen/unseen composition analysis, and the three anti-memorization
slices. Five seeds are used for claim-bearing runs, with mean, standard deviation, 95% CI, and paired
bootstrap tests.

## Decision criteria

The machine-readable criteria live in `research/experiments/experiment_registry.yaml`. GO requires
positive OOD graph advantage, higher graph retention, increasing advantage with depth, stability
across seeds, and failure of random structure to reproduce the gain. A reference backend can validate
the instrument but cannot produce a scientific GO or NO-GO decision.
