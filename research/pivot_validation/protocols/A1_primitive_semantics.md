# A1 — Primitive Semantic Availability

## Scientific Question

Can frozen LLaVA make direct image-only decisions about object shape, object color, and one-hop
spatial relation on the existing synthetic scenes?

## Hypothesis

H-A predicts that at least one primitive dimension fails the preregistered adequacy gate. The test
is behavioral and decision-facing; it does not claim to inventory every hidden representation.

## Variables

- Independent variable: primitive task (`object_shape`, `attribute_color`, `direct_relation`).
- Dependent variable: four-choice accuracy and chance-normalized accuracy.
- Image evidence: unchanged frozen scene image.

## Controlled Factors

Each scene produces one question per task. Shape questions identify a unique target without naming
its shape; color questions identify a unique target without naming its color; relation questions use
one declared direct edge. Every question has four deterministic choices, balanced answer positions,
and no textual evidence beyond `image only`.

## Evaluation Metrics

Report per-task accuracy, macro semantic score, chance-normalized score, per-seed values, relation
slices, and answer-position balance. Chance is 0.25 for all tasks.

## Statistical Protocol

The first three seeds form a pilot status only. Final classification requires five seeds. Report
95% percentile-bootstrap intervals over seed accuracies and paired scene-level descriptive counts.

## GO / NO-GO Criteria

A dimension is `HIGH` when mean accuracy is at least 0.70, the seed-bootstrap lower bound is at
least 0.60, and at least 80% of seed accuracies exceed chance. It is `LOW` when mean accuracy is at
most 0.60 and the upper bound is at most 0.70. Other outcomes are `INDETERMINATE`.

H-A is falsified only if all three dimensions are `HIGH`. Any `LOW` dimension supports a scoped
semantic-availability limitation; indeterminate dimensions prevent a clean classification.
