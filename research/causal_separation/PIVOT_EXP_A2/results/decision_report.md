# PIVOT_EXP_A2 Conditional Causal Effects Report

## Scientific adjudication

**INCONCLUSIVE**

At least one preregistered global or manipulation gate failed.

## Why the study is inconclusive

- Semantic sufficiency failed: oracle accuracy was 0.7275 with 95% CI [0.6725, 0.7800], below the
  registered mean threshold of 0.90 and CI-lower threshold of 0.85.
- Parsing integrity failed: 137 of 5,300 outputs did not match an allowed answer under the frozen
  evaluator. All 137 were manipulation-check trials; none were deleted, re-run, or reclassified.
- Prediction count, five-seed completeness, eight-cell matrices, token matching, corruption
  validity, freeze hashes, and the artifact manifest passed.

## Semantic rescue validation

- Oracle accuracy: 0.7275
- Corrupted accuracy: 0.0425
- Paired rescue gain: 0.6850
- Manipulation passed: False

## Primary conditional effects

| Estimand | Mean | 95% CI |
| --- | ---: | --- |
| E1_semantic_rescue | -0.0061 | [-0.0317, 0.0178] |
| E2_relation_under_oracle | 0.3156 | [0.2744, 0.3567] |
| E3_json_minus_triples | 0.0778 | [0.0333, 0.1289] |
| E4_semantic_x_relation | 0.0344 | [0.0067, 0.0611] |
| E5_semantic_x_format | -0.0311 | [-0.0844, 0.0200] |
| E6_three_way | -0.0689 | [-0.1356, -0.0111] |

These estimands are descriptive diagnostics only. Because semantic sufficiency and parsing
integrity failed, they cannot identify a semantic-primary, integration-independent, or joint causal
mechanism.

## Next-stage authorization

```yaml
independent_replication_allowed: false
model_development_allowed: false
paper_writing_allowed: false
```

This LLaVA-only result is not a general claim about modern VLMs.
