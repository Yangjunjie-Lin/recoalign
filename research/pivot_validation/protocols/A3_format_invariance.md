# A3 — Representation Format Invariance

## Scientific Question

Does frozen LLaVA use the same bound relation facts invariantly when they are serialized as natural
language, canonical triples, or JSON?

## Hypothesis

H-B predicts representation-format sensitivity even after semantic facts and model-token budgets
are matched.

## Variables

- Natural language relation statements.
- Canonical `(subject, relation, object)` triples.
- Compact JSON relation objects.
- Dependent variable: unchanged compositional-question accuracy.

## Controlled Factors

All formats use the same query-role aliases and identical canonical relation-fact hash. No object or
attribute facts are added. The same image, question, choices, relation order, model, and decoding are
used. Repeated punctuation-only `.` padding units match complete prompt length within one actual
LLaVA token; natural and matched counts remain available for audit.

## Evaluation Metrics

Report per-format accuracy, all paired differences, seed-bootstrap confidence intervals, token
counts, fact hashes, question-type slices, and hop-depth slices.

## Statistical Protocol

Final decisions require five seeds. Format equivalence requires every pairwise difference interval
to lie inside ±0.05. Format sensitivity requires at least one absolute mean difference of 0.05,
a confidence interval excluding zero, and the same direction in at least 80% of seeds.

## GO / NO-GO Criteria

`FORMAT_SENSITIVE` supports H-B only as a diagnostic fact; it does not identify a remedy.
`FORMAT_INVARIANT` falsifies the format component when all pairwise equivalence gates pass.
Mixed or underpowered comparisons are `INCONCLUSIVE` and block a unique mechanism decision.
