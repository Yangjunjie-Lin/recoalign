# Scientific decision protocol

The decision engine implements the experiment registry exactly as committed. It does not select the
best seed, add a favorable metric, tune a threshold, or reinterpret a failed integrity gate.

## Inputs

The engine consumes a schema-valid governed result and its experiment registration. For each primary
criterion it reads the cross-seed effect, 95% confidence interval, preregistered significance test,
number of valid seeds, and fraction of seeds with the predicted direction. Dataset-manifest assertions
are conjunctive integrity gates.

## Decision semantics

- **GO**: every effect-size, confidence, significance, seed-count, robustness, and integrity gate
  passes.
- **NO-GO**: execution and provenance are complete, but at least one registered gate fails.
- **INCONCLUSIVE**: a required aggregate, test, integrity record, or eligible model is missing; no scientific decision
  is permitted.

Model eligibility is a mandatory gate. The deterministic reference backend validates infrastructure
only and therefore yields INCONCLUSIVE even if its synthetic condition scores satisfy numerical gates.

The three initial protocols use directional one-sided tests because their direction is preregistered.
Alpha is 0.05, confidence is 95%, the default is three seeds, and experiments marked critical require
five. EXP002 has two conjunctive primary comparisons; both must pass. No implicit multiple-testing
exception is allowed.

## Statistical methods

- Paired t-test: paired scene effects and preregistered cross-seed effects where its assumptions are
  accepted.
- Percentile bootstrap: default 95% uncertainty interval with a configuration-controlled resampling
  seed and sample count.
- Paired permutation test: deterministic exact sign flips for small samples, seeded Monte Carlo for
  larger samples when distributional assumptions are doubtful.

The implementation is in `evaluation/statistics/`. Statistical inputs reject booleans, non-numeric
values, non-finite values, unmatched pairs, and empty samples.

## Audit artifact

Governed runs generate an authoritative, schema-valid `decision_report.yaml` and a human-readable
`decision_report.md` mirror. The YAML record contains `experiment_id`, `evidence`,
`statistical_result`, `robustness_check`, `final_decision`, and `next_action`. Its final decision is
exactly one of GO, NO-GO, or INCONCLUSIVE. Reapply the committed rule with:

```bash
recoalign decide-experiment runs/<experiment-id>/<run-id>/metrics.json
```

A reviewed decision copied to `research/decisions/` must retain the run ID and hashes. Changing a rule
requires a new preregistration and cannot rewrite an existing decision.
