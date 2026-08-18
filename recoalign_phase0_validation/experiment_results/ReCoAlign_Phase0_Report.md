# ReCoAlign Phase 0 Report

Generated: 2026-08-18T00:14:20.818757+00:00

## 1. Experimental Setup

This run tests whether a frozen cross-modal projection selectively removes semantic structure that is linearly decodable from its frozen vision input. The dataset contains 10,000 main images (8,000 train / 2,000 test) and 500 relation-isolation pairs. The frozen backend is `llava_1_5_7b_vision_projector` using `liuhaotian/llava-v1.5-7b`. `Zv` and `Za` are captured at the representation boundaries documented in `run_config.json`; neither the encoder nor the projection is trained or modified. Only logistic-regression probes are fitted.

All randomness uses seed `20260818`. Accuracy uncertainty is estimated with 2,000 paired bootstrap resamples. Standard deviations and confidence intervals are in `results.csv`; per-sample predictions are retained under `predictions/`.

Protocol validity: **FULL**.

## 2. Results

| Semantic | Zv accuracy | Za accuracy | SPD (Zv − Za) | 95% CI of drop |
| --- | ---: | ---: | ---: | ---: |
| Object | 100.00% | 100.00% | 0.00% | [0.00%, 0.00%] |
| Attribute | 99.95% | 100.00% | -0.05% | [-0.15%, 0.00%] |
| Relation | 100.00% | 99.90% | 0.10% | [0.00%, 0.25%] |
| Composition | 100.00% | 100.00% | 0.00% | [0.00%, 0.00%] |

The paired bootstrap estimate for `Relation drop − Object drop` is 0.10%, 95% CI [0.00%, 0.25%], one-sided p = 0.145927.

Semantic-isolation relation accuracy is 99.90% at Zv and 99.90% at Za (drop 0.00%, 95% CI [0.00%, 0.00%]). Pair-level both-correct accuracy is 99.80% at Zv and 99.80% at Za.

## 3. Evidence for Semantic Preservation Failure

Observed pattern: **Pattern C: no material degradation at the tested projection**.

Pre-registered decision gates:

- `object_drop_lt_10pp`: PASS
- `attribute_drop_lt_15pp`: PASS
- `relation_drop_gt_25pp`: FAIL
- `composition_drop_gt_25pp`: FAIL
- `relation_vs_object_p_lt_0_05`: FAIL
- `isolation_drop_ci_above_zero`: FAIL
- `full_protocol`: PASS

The experiment calls selective preservation failure only when the exact asymmetric thresholds are met, the Relation-vs-Object drop is significant, the relation-isolation control has a drop whose 95% CI excludes zero, and the full protocol was run.

## 4. Alternative Explanation Analysis

- **Relation is intrinsically harder:** absolute Relation accuracy may be lower, but SPD compares the same task before and after projection. The paired isolation set strengthens this control by holding objects, attributes, rendering nuisance, and pair identity fixed while changing only the relation.
- **Generic alignment bottleneck:** similar drops across all four factors are classified as Pattern B and produce NO-GO. Only the asymmetric Pattern A can pass.
- **Probe learns the task:** every probe is a single logistic-regression layer with fixed regularization; no nonlinear probe, VLM fine-tuning, or learned representation is used.
- **Ceiling effect / synthetic task too easy:** near-perfect accuracy at both stages limits the experiment's ability to expose subtle selective loss. This is a reason to retain NO-GO rather than reinterpret a null result as proof that all VLM semantics are preserved.
- **Synthetic-world scope:** a positive result would establish a controlled mechanism, not prevalence in natural imagery. A negative result only rejects this mechanism for the tested projection and synthetic distribution; it does not prove all VLM stages preserve semantics.
- **LLM-stage scope:** this Phase 0 path intentionally stops after the alignment projector and does not load the 7B language model, so `Zl` is unavailable. The run tests the registered `Zv → Za` mechanism only; it makes no claim about losses inside LLM layers.

## 5. Final Decision

# NO-GO

The experiment does not satisfy every pre-registered selective-degradation gate. ReCoAlign is not advanced on the basis of this run; no favorable post-hoc interpretation is substituted.
