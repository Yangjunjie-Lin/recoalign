# Experiment Name

## Scientific Question

State the single falsifiable scientific question. Explain which causal or mechanistic claim the
experiment can test and which stronger claims it cannot establish.

## Hypothesis

- Registry hypothesis ID:
- If the hypothesis is true, the preregistered observation is:
- Falsification condition:

## Variables

- Independent variables: controlled condition(s) intentionally varied.
- Dependent variables: preregistered primary and secondary measurements.

## Controlled Factors

List the model/checkpoint, examples, prompts, decoding, preprocessing, dataset version, evaluation
implementation, and every other factor held fixed across conditions.

## Conditions

- Baseline:
- Comparison:
- Ablation/negative control:
- Information-equivalence argument:

## Evaluation Metrics

Name every metric, its direction, unit, aggregation, and scientific rationale. Exactly one metric (or
an explicitly conjunctive set) must be designated primary before results are inspected.

## Statistical Protocol

- Random seeds: 3 by default; 5 for experiments marked critical in the registry.
- Confidence interval: 95% percentile bootstrap interval unless preregistered otherwise.
- Significance test: paired t-test for approximately continuous paired effects; paired bootstrap for
  uncertainty; paired permutation test when distributional assumptions are doubtful.
- Significance level and alternative hypothesis:
- Multiple-comparison correction, if applicable:
- Missing/failed seed handling: retain and report failures; do not silently replace seeds.

## GO / NO-GO Criteria

- GO: all preregistered effect-size, confidence, significance, robustness, and integrity gates pass.
- NO-GO: any required gate fails. Do not add post-hoc metrics to reverse the decision.
- INCONCLUSIVE: execution, provenance, or evidence eligibility is insufficient for a decision.

Benchmark-only experiments without a registered, falsifiable hypothesis are prohibited.

## Reproducibility Binding

Record experiment ID, hypothesis ID, code commit/dirty state, resolved YAML, dataset manifest/version,
model/checkpoint identity, command, environment, every seed, metrics JSON, logs, and run manifest.
