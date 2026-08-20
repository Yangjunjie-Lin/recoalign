# Decision Report — EXP002

- Hypothesis: `H002`
- Run: `exp002-20260820T144451.434722+0000`
- Evidence role: `scientific_evidence`
- Final decision: **GO**
- Next action: Proceed to independent replication and reviewed evidence promotion.

## Statistical result

| Metric | Mean | Minimum | CI | p-value | Pass |
|---|---:|---:|---:|---:|:---:|
| `structural_dependency.full_over_partial_25.estimate` | 0.29 | 0.01 | [0.27666666666666667, 0.30333333333333334] | 9.999000099990002e-05 | YES |
| `structural_dependency.full_over_partial_50.estimate` | 0.30166666666666664 | 0.03 | [0.28500000000000003, 0.31999999999999995] | 9.999000099990002e-05 | YES |
| `structural_dependency.full_over_partial_75.estimate` | 0.355 | 0.05 | [0.33666666666666667, 0.37666666666666665] | 9.999000099990002e-05 | YES |
| `structural_dependency.full_over_random_graph.estimate` | 0.45166666666666666 | 0.05 | [0.44000000000000006, 0.4616666666666666] | 9.999000099990002e-05 | YES |
| `structural_dependency.full_over_wrong_graph.estimate` | 0.18 | 0.05 | [0.15833333333333333, 0.2033333333333333] | 9.999000099990002e-05 | YES |
| `structural_dependency.full_over_relation_label_randomized.estimate` | 0.3566666666666667 | 0.05 | [0.3416666666666667, 0.38] | 9.999000099990002e-05 | YES |
| `depth_dependency.full_over_random_graph.slope` | 0.138 | 0.0 | [0.11200000000000002, 0.16733333333333336] | 9.999000099990002e-05 | YES |

## Robustness check

- Result complete: True
- Registry validated: True
- Model matches registration: True
- Dataset matches registration: True
- Model eligible for scientific decision: True
- Overall robustness pass: True

This Markdown file mirrors `decision_report.yaml`, which is the authoritative record.
Threshold changes require a new registration before rerunning.
