# PIVOT_EXP_A3 Legacy Parse-Failure Audit

## Immutable scope

This is a read-only format audit of the 137 frozen PIVOT_EXP_A2 manipulation-check rows whose
frozen `evaluation.evaluation_method` is `no_match`. The source SHA-256 is
`5d31a46bb6bad7630d5dfd4095195aae3fe6b6b69972eadccbd95c1c846d886c`. No prediction was changed, deleted, reinterpreted, or
rescored, and this audit cannot be used to recalculate PIVOT_EXP_A2 accuracy or alter its
`INCONCLUSIVE` decision.

## Frozen selection

- Source predictions: 5300
- Selected `manipulation_check` / `no_match` rows: 137
- Frozen boundary passed: true

## Distribution

- By task: `{"color": 18, "entity_attribute_binding": 118, "shape": 1}`
- By condition: `{"corrupted_semantics": 73, "oracle_semantics": 64}`
- By raw-output pattern: `{"hallucinated_choice": 19, "multiple_choices": 118}`
- At least one registered choice string appears: 118
- Multiple registered choices appear: 118
- Alias variants appear: 0
- Registered answer string appears inside an invalid output:
  105
- Genuinely off-task: 0

The last-but-one flag is string-occurrence evidence only. It does not convert a multiple-choice
list into a valid answer and is not a revised correctness judgment.

## Measurement implication for PIVOT_EXP_A3

The dominant failure is multi-option emission rather than an empty response: the old free-text
instrument mixed answer selection with response formatting. PIVOT_EXP_A3 therefore uses
conditional-likelihood forced choice as its primary measurement and a frozen exact grammar
(`FINAL_CHOICE=<1|2|3|4>`) only as a secondary external-validity measure. Numeric option IDs avoid
the historical `Entity A/B/C` alias namespace.
