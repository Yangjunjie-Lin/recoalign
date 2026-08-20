# Composition and factor-holdout split protocol

## Split unit

A composition signature is computed from the query subject factors, primary relation, query target
factors, and question family. The factors are color, shape, size, texture, and category. Assignment
is deterministic and never depends on a model prediction.

## IID split

The stable SHA-256 bucket of the sample ID assigns 80% train, 10% validation, and 10% test. This is a
pipeline sanity check, not evidence of compositional generalization.

## Composition split

For each unordered endpoint-factor pair and relation, one lexical orientation is assigned to train
and the reversed orientation to test. A stable 10% of unordered families is reserved for validation.
This operationalizes the intended contrast:

```text
train: red cube left blue sphere
test:  red sphere left blue cube
```

Full ordered composition signatures must have zero train/test intersection. Validation reports the
exact overlap set and fails closed when it is non-empty. Known primitive vocabularies remain shared;
the intervention is their binding/order, not the introduction of a new token.

## Relation split

`left` and `right` are train relations. `front` and `behind` are held out for test. Other relations
are validation-only under this policy. This split asks whether a reasoning interface transfers to a
relation family absent from training; it must not be described as pure recombination of seen
relations.

## Attribute split

The query-subject color controls assignment: red/blue are train, green is test, and yellow/purple are
validation. Other factors and relation templates remain available in all partitions. Leakage checks
operate on the query subject rather than every distractor object, so a test-world distractor cannot
silently change the intervention definition.

## Reporting requirements

Every run reports sample counts per partition, primitive sets, ordered composition overlap, question
family distribution, relation distribution, and hop-depth distribution. Empty required partitions,
post-hoc sample movement, and target-model-dependent filtering invalidate the run. EXP003 must bind
its chosen split policy and generator version in the experiment registry before scientific use.
