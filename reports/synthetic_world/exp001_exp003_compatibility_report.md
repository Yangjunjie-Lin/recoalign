# EXP001 / EXP002 / EXP003 Compatibility Report

## EXP001 — Graph vs Text

The generator produces `image`, `object_list`, `caption`, and `scene_graph` for the same sample ID,
world state, question, and answer. Caption and scene graph are generated from the identical ordered
canonical fact set. `caption_facts_sha256 == graph_facts_sha256` is schema-enforced and recomputed by
validation, blocking the “graph contains extra information” confound.

The evaluator exposes the existing runtime condition names:

```text
image_only, object_list, caption, scene_graph
```

Paired predictions retain sample ID, split, composition, seed, question family, relation, and hop
depth for paired contrasts and slice analysis.

## EXP002 — Graph ablation

The same sample can be evaluated as:

```text
scene_graph, partial_graph, random_graph, corrupted_graph
```

`partial_graph` removes one required edge (and removes the sole edge for a one-hop sample).
`corrupted_graph` deterministically substitutes one relation. All variants preserve object nodes,
image, question, choices, and scoring key. The image is retained for every ablation condition, so
edge corruption is the only evidence intervention.

## EXP003 — OOD composition

The v2 split planner supports four named policies: IID, composition, relation, and attribute. The
composition policy reports exact train/test ordered-signature overlap and fails if it is non-empty.
Relation and attribute policies report their primitive sets. This directly supports the OOD
composition contrast while separating it from the stronger relation-primitive and attribute-
primitive holdouts.

## Governance binding

EXP001 was explicitly amended by the controlled-validation implementation to bind
`generator-v2` and `manifests/datasets/synthetic_world_v2.yaml`, together with token, depth, and OOD
decision gates. EXP002/EXP003 retain their earlier registrations until their own protocol amendments.
The condition API remains backward compatible.

## Reviewer-risk closure

For every graph-vs-caption pair, the semantic fact inventory is equal by construction and checked by
hash. Renderer metadata, oracle provenance, supporting edges, and split signatures are stored per
sample. Therefore a graph advantage cannot be attributed to additional declared facts; it remains a
testable effect of the structured interface representation.
