# Scientific Adjudication of PIVOT_EXP_A

## Scope

This adjudication changes no prediction, metric, threshold, seed, dataset, checkpoint, protocol, or
frozen implementation. It resolves a post-run decision-logic conflict by applying the registered
multiple-explanations rule to the preserved five-seed metrics.

## Conflict

The automated classifier returned `GO / PRIMITIVE_SEMANTIC_REPRESENTATION_LIMITATION`. Its first
eligible branch uses LOW A1 primitive scores and does not test whether A3 simultaneously reports a
stable format effect. Final evidence does report both:

- H-A evidence: shape, color, and direct-relation behavior are all LOW.
- H-B evidence: relation and complete evidence contributions are stable, integration surplus is
  stable, and all A3 format pairs are sensitive across five seeds.

Therefore H-B cannot be recorded as `INCONCLUSIVE`, and H-A cannot be selected as the unique primary
mechanism from these tests.

## Ruling

The formal decision is `NO-GO / MULTIPLE_SUPPORTED_MECHANISMS_CAUSAL_PRIMACY_UNRESOLVED`.

PH001 survives only as a joint descriptive phenotype: limited primitive-semantic behavior coexists
with format-dependent relational-evidence use. A unique causal bottleneck has not been isolated.
Model development and paper writing remain blocked.
