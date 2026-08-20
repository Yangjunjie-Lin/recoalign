# Structural Necessity Analysis

- Final governed decision: **GO**
- Backend: `llava_1_5_7b`
- Structural Dependency Score (SDS) is Full accuracy minus the named control.

| Registered effect | Mean SDS | 95% CI | p-value |
|---|---:|---:|---:|
| `structural_dependency.full_over_partial_25.estimate` | 0.2900 | [0.2767, 0.3033] | 9.999e-05 |
| `structural_dependency.full_over_partial_50.estimate` | 0.3017 | [0.2850, 0.3200] | 9.999e-05 |
| `structural_dependency.full_over_partial_75.estimate` | 0.3550 | [0.3367, 0.3767] | 9.999e-05 |
| `structural_dependency.full_over_random_graph.estimate` | 0.4517 | [0.4400, 0.4617] | 9.999e-05 |
| `structural_dependency.full_over_wrong_graph.estimate` | 0.1800 | [0.1583, 0.2033] | 9.999e-05 |
| `structural_dependency.full_over_relation_label_randomized.estimate` | 0.3567 | [0.3417, 0.3800] | 9.999e-05 |
| `depth_dependency.full_over_random_graph.slope` | 0.1380 | [0.1120, 0.1673] | 9.999e-05 |

Reference-backend results are infrastructure validation only and cannot establish the mechanism claim.
