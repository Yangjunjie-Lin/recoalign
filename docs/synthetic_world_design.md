# Controlled Synthetic World design

## Scientific purpose

The ReCoAlign Synthetic Compositional World is a scientific instrument for testing the Structured
Reasoning Interface Gap. It is not intended to approximate a natural-image distribution or maximize
a public leaderboard score. Its causal question is narrower:

> When visual semantics are fixed, does changing only the representation interface change a frozen
> model's ability to execute the same compositional reasoning problem?

The benchmark follows one directional construction:

```text
controlled world state
  -> independently declared semantic factors
  -> oracle typed relations
  -> derived image / graph / caption
  -> programmatically derived compositional QA
  -> IID and held-out-factor evaluation
```

An image, caption, or answer is never sampled first. No LLM produces captions or labels.

## Factorized world state

Each object is the Cartesian product of finite, versioned factors:

| Factor | Values |
| --- | --- |
| shape | circle, square, triangle, cube, sphere |
| category | animal, vehicle, object |
| color | red, blue, green, yellow, purple |
| size | small, medium, large |
| texture | solid, striped, dotted |

Shape and color use independently shuffled streams with a per-scene uniqueness constraint so every
textual reference has exactly one referent. Category, size, and texture are sampled from independent
streams. This constraint is recorded because exact reference resolution is more important than
claiming unconstrained empirical independence.

Relations use one canonical triple representation:

```json
{"subject": "obj1", "relation": "left", "object": "obj2"}
```

The declared vocabulary is spatial (`left`, `right`, `above`, `below`), depth (`front`, `behind`),
distance (`near`, `far`), containment (`inside`, `contains`), and interaction (`touching`, `holding`).

## Licensed inference rules

Ground-truth inference is deliberately smaller than everyday-language inference:

- `left`, `right`, `above`, `below`, `front`, and `behind` license same-relation transitivity and are
  used for 2–4 hop questions.
- `inside` and `contains` are represented as inverse containment relations, but current generated
  QA uses them only at one hop.
- `near`, `far`, and `touching` are symmetric only for direct pairs. They are not transitive.
- `holding` is directed and one hop.

Mixed-axis paths do not yield a single answer. For example, `A left B` and `B behind C` do not imply
that A is behind C. Rejecting such questions prevents nominal hop depth from introducing invalid
labels.

## Derived views and information control

Every sample stores the object inventory, appearance attributes, canonical relations, oracle scene
graph, deterministic caption, image, question, answer, choices, split, and reconstruction metadata.
Graph and caption are derived from the same ordered canonical fact set. The sample stores:

- `canonical_facts_sha256`;
- `information_control.graph_facts_sha256`;
- `information_control.caption_facts_sha256`;
- `information_control.caption_graph_equivalent=true`.

Validation regenerates the caption and both hashes from the world state. A hand-edited caption or a
graph with extra facts fails validation. The unordered object-list condition is an explicit
object-semantic control; it is not claimed to contain relational facts.

## Question families and complexity

Four deterministic template families are generated: object reasoning, relation reasoning,
attribute reasoning, and multi-hop relation reasoning. The answer is recomputed from the graph or
queried object attribute during validation. Supporting edge IDs are stored in `metadata.query`.
Difficulty is defined as the number of necessary supporting edges, not sentence length. Values 1,
2, 3, and 4 are supported.

## Rendering and reconstruction

`DeterministicRenderer` uses only the declared objects, relations, resolution, and style. Styles are
`flat`, `outline`, and `pastel`. Category markers, texture patterns, size, depth scaling, and
containment layout are deterministic. Each materialized sample contains `image.png`, `scene.json`,
`graph.json`, and `metadata.json`; the aggregate image SHA-256 is stored in the record. Rebuilding a
sample must reproduce identical PNG bytes.

## Shortcut controls

- Relation direction, object factors, question family, and hop depth are generated independently of
  any target-model response and reported as slices.
- Object IDs are graph handles and are not rendered into the image; questions use semantic
  descriptions with unique referents.
- Caption and graph carry the same facts, while partial/corrupted controls preserve image, nodes,
  question, and answer key.
- Composition assignment is a deterministic factor rule, not a random post-hoc selection.
- The full 12-relation distribution is retained; samples are never filtered to improve accuracy.
- Resolution and style are recorded intervention variables, enabling style-shift checks without
  changing world semantics.

Synthetic geometry necessarily makes spatial relations visually regular. Claims must therefore be
reported by relation, factor, question family, hop depth, and renderer style rather than relying only
on aggregate accuracy.

## Falsifiability and known boundaries

The instrument can falsify an interface claim when graph and caption carry identical facts but a
pre-registered frozen model shows no robust paired graph advantage. It cannot by itself establish
natural-image generality, perceptual graph extraction quality, or a learned ReCoAlign mechanism.
Those require later experiments. Renderer artifacts are therefore reported as a limitation and
must not be optimized in response to target-model accuracy.
