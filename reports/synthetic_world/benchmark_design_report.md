# Synthetic Benchmark Design Report

## Outcome

ReCoAlign now has a world-first, deterministic compositional benchmark implementation at
`src/recoalign/synthetic_world/`. The implementation treats the synthetic world as an intervention
instrument for the Structured Reasoning Interface Gap, not as target-model-tuned training data.

## Initial implementation audit

The repository audit found a useful v1 smoke-test path but not a controlled benchmark: it generated a
fixed three-object, same-relation chain over six directional/depth relations; rendering was tied to a
single relation layout; object size/texture, containment, distance, and interaction were absent; the
schema lacked caption, attributes, split, and information-control provenance; OOD logic only deduped
a composition string; and there was no dedicated split validator, question validator, multi-seed
CLI, or `tests/synthetic_world/` suite. The root v1 import paths are retained as compatibility shims,
while all new authority lives under `src/recoalign/synthetic_world/`.

## Ontology

- Objects factor into shape (5), category (3), color (5), size (3), and texture (3).
- Relations cover spatial (4), depth (2), distance (2), containment (2), and interaction (2).
- Every relation is a canonical `(subject, relation, object)` triple over declared object IDs.
- Inference rules explicitly distinguish transitive, inverse, symmetric, and directed relations.

## Generation process

1. Derive a sample seed from dataset seed and stable index.
2. Generate object factors and a licensed relation chain as world state.
3. Construct and validate the oracle scene graph.
4. Derive an unordered object inventory and lossless caption from canonical facts.
5. Derive a question, answer, choices, query endpoints, and supporting edge IDs.
6. Assign a deterministic IID or held-out-factor split.
7. Render the world from declared state, resolution, and style.
8. Validate schema, ontology, answer, equivalence hashes, leakage, and image checksum.

No answer or caption is LLM-generated. Scene graphs are ground truth and are never model outputs.

## Split strategy

- IID: stable 80/10/10 hash partition.
- Composition: endpoint-factor orientation is held out while primitives remain known.
- Relation: left/right train and front/behind test; other relations are validation controls.
- Attribute: red/blue query subjects train, green test, yellow/purple validation.

The leakage validator fails on ordered composition overlap or violation of relation/attribute
policies. Full definitions are in `docs/composition_split_protocol.md`.

## Falsification controls

Caption and graph expose the same canonical object and relation facts, proven by independently stored
fact hashes and deterministic regeneration. Partial and corrupted graphs change only structured
edges while preserving objects, image, question, and answer. Multi-hop labels are generated only for
relations whose transitivity is licensed by the symbolic semantics.

## Scope boundary

The benchmark establishes controlled mechanism evidence. It does not claim natural-image validity,
learned graph extraction, or state-of-the-art benchmark performance. Any such claim requires a
separate preregistered experiment.
