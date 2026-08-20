# Decision Report — EXP003

- Hypothesis: `H003`
- Run: `exp003-failed-20260820T155940.441157+0000`
- Evidence role: `scientific_evidence`
- Final decision: **INCONCLUSIVE**
- Next action: Resolve missing or ineligible evidence and rerun the preregistered protocol.

## Statistical result

| Metric | Mean | Minimum | CI | p-value | Pass |
|---|---:|---:|---:|---:|:---:|
| `ood.graph_over_caption.estimate` | None | 0.05 | missing | None | NO |
| `generalization.retention_advantage.graph_over_caption` | None | 0.02 | missing | None | NO |
| `reasoning_depth.graph_advantage_slope` | None | 0.0 | missing | None | NO |
| `ood.graph_over_random_graph.estimate` | None | 0.05 | missing | None | NO |

## Robustness check

- Result complete: False
- Registry validated: True
- Model matches registration: True
- Dataset matches registration: True
- Model eligible for scientific decision: True
- Overall robustness pass: False

This Markdown file mirrors `decision_report.yaml`, which is the authoritative record.
Threshold changes require a new registration before rerunning.
