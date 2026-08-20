# EXP002 — Graph Completeness and Structural Necessity

## Scientific Question

Does VLM reasoning depend on the correctness and completeness of structural relations, rather than
on graph-like prompt appearance or graph length?

## Hypothesis

H002, Structural Necessity: if graphs act as a reasoning interface, deleting or semantically
corrupting their relations will cause predictable performance degradation, and degradation will
increase with the number of reasoning hops.

## Variables

- Independent variable: image-only, full graph, controlled 25/50/75% relation removal, relation
  flip, entity swap, same-size random graph, serialization order, or opaque relation labels.
- Dependent variables: accuracy and paired Structural Dependency Score (SDS), defined as
  `accuracy(full) - accuracy(control)` on identical scenes.
- Diagnostic variables: hop depth (1–4) and relation family (spatial, depth, containment).

## Controlled Factors

The image, question, answer choices, object inventory, checkpoint, decoding path, dataset seed, and
prompt template are held fixed. Relation flip, entity swap, and random graph preserve edge count.
Full and Random prompts are padded with content-free sentinels until the model tokenizer reports an
exact zero-token difference. The randomizer receives only the graph and seed, never the answer.

## Conditions

- C0 Image Only: image and unchanged question, with no explicit graph.
- C1 Full Graph: complete ground-truth nodes and relations.
- C2 Partial Graph: supporting relations removed at preregistered ratios 25%, 50%, and 75%.
- C3 Corrupted Graph: relation flip, entity swap, and random graph are retained separately; no
  corruption result may be selected or hidden after inspection.

Anti-shortcut controls randomize edge serialization order, replace predicates with undisclosed
opaque labels, and exactly token-match Full versus Random.

## Evaluation Metrics

Report condition accuracy, every SDS (not only the largest), performance versus removal ratio,
1/2/3/4-hop slices, depth slope, exact relation and grouped relation tables, token deltas, and all
corruption invariant checks. The manifest records the original graph, intervention, output graph,
seed, ratio, selected edges, hashes, and preserved invariants for every sample.

## Statistical Protocol

The paper protocol uses the five committed seeds; development runs require at least three. All
primary contrasts use identical scenes and a one-sided centered paired bootstrap with 10,000 draws.
Across seeds report mean, sample standard deviation, percentile-bootstrap 95% CI, paired-bootstrap
p-value, individual values, and positive-seed fraction. Alpha is 0.05.

## GO / NO-GO Criteria

GO requires Full to outperform all three Partial ratios, Random, the mean Wrong Graph control, and
opaque relation labels; every registered interval must exclude zero, every paired-bootstrap p-value
must be at most 0.05, at least four of five seed effects must be positive, and the Full-minus-Random
effect must have a positive depth slope. The corruption, leakage, and exact length manifests must
all pass. Random approximately matching Full, Partial having no effect, Wrong Graph having no effect,
or an effect confined to one-hop tasks is NO-GO for an eligible target VLM. Missing/ineligible model
evidence or failed integrity checks is INCONCLUSIVE and cannot be promoted to a mechanism claim.
