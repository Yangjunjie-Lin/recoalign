# EXP003 — Structured Interface for OOD Compositional Generalization

## Scientific Question

Does a structured intermediate representation allow a frozen VLM to reason over unseen
visual-semantic compositions rather than match graph patterns observed in the seen partition?

## Hypothesis

H003 predicts that graph evidence will retain more IID performance than an information-equivalent
caption on controlled OOD tests. H003 is falsified for this protocol if graph advantage disappears
on OOD, caption retention matches graph retention, advantage does not grow with reasoning depth, or
a length-matched random graph reproduces the gain.

## Variables

Independent variables are interface condition, IID/OOD split family, seen/unseen partition, and hop
depth. Dependent variables are accuracy, paired graph advantage, generalization gap, OOD retention,
and depth trend.

## Controlled Factors

The frozen checkpoint, prompts, choices, decoding, generator, primitive vocabulary, render style,
sample counts, test membership, and statistics are fixed across conditions and seeds. Graph and
caption encode the same canonical facts; correct and random graphs have matched node/edge counts and
input length.

## Constructive split policy

Membership is determined before model execution; test predictions never affect allocation.

- `iid`: test compositions occur in train while size, texture, and category nuisance factors change.
- `composition_ood`: colors, shapes, and relation primitives occur in train, but ordered endpoint
  color-shape bindings and complete relation programs in test do not.
- `relation_ood`: train and test use the same atomic relations at fixed depth two; ordered relation
  pairs such as `left+behind` are test-only.
- `hop_ood`: train contains one- and two-hop chains; test contains three- and four-hop chains.

Every partition contains images, oracle graphs, deterministic lossless captions, questions, and
answers. Per-file SHA-256 hashes and semantic fact hashes are recorded. Validation fails closed on
sample ID overlap, composition/relation-program/depth leakage, missing training primitives, invalid
answers, or hash mismatch.

## Frozen evaluation protocol

No training, parameter change, OOD-specific prompt, or split-specific decoding is allowed. The same
question template and decoding rule are applied under four paired conditions:

1. image only;
2. image plus caption;
3. image plus correct scene graph;
4. image plus deterministic random graph with nodes, edge count, and input length controlled.

Seen partitions are scored only to estimate IID references, seen-composition effects, and the
one/two-hop section of the depth curve. All OOD claims use held-out test partitions.

## Evaluation Metrics

The report includes overall and per-split accuracy, IID minus OOD generalization gap, OOD/IID
retention, one-through-four-hop curves, seen/unseen composition tables, and object-identity swap,
relation-recombination, and attribute-transfer slices. Graph-caption and graph-random contrasts use
paired scene-level bootstrap. IID/OOD gaps use deterministic template-index pairing. Five seeds are
required for the paper protocol (three for non-critical development), with mean, standard deviation,
and percentile-bootstrap 95% confidence intervals.

## Statistical Protocol

All interface contrasts are paired by scene. IID/OOD gaps are paired by predeclared template index.
The paper run uses five independent seeds, 10,000 fixed-seed bootstrap samples, mean, standard
deviation, percentile 95% confidence intervals, and one-sided paired-bootstrap tests.

## GO / NO-GO Criteria

GO requires all of the following across five stable seeds:

- OOD graph-over-caption accuracy gain at least 0.05;
- graph retention minus caption retention at least 0.02;
- a positive graph-advantage slope with reasoning depth;
- OOD graph-over-random-graph gain at least 0.05;
- lower 95% bounds above zero, paired-bootstrap p-values at most 0.05, at least 80% positive seeds,
  and every manifest assertion passing.

Any completed eligible-model run that fails a registered gate is NO-GO. Reference-backend runs are
infrastructure validation and remain INCONCLUSIVE regardless of their synthetic condition rates.
