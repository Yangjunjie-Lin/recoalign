# Decision Report — EXP001

- Hypothesis: `H001`
- Run: `exp001-20260820T134152.769731+0000`
- Evidence role: `scientific_evidence`
- Final decision: **NO-GO**
- Next action: Record the falsifying result and revise or retire the hypothesis.

## Statistical result

| Metric | Mean | Minimum | CI | p-value | Pass |
|---|---:|---:|---:|---:|:---:|
| `contrasts.contrasts.graph_over_text.estimate` | -0.043333333333333335 | 0.05 | [-0.058333333333333334, -0.02833333333333333] | 0.9965063126673401 | NO |
| `controlled_contrasts.token_matched.graph_over_caption.estimate` | -0.07833333333333334 | 0.05 | [-0.10166666666666666, -0.05500000000000001] | 0.9979032872998019 | NO |
| `reasoning_depth.token_matched.2.graph_over_caption.estimate` | 0.0 | 0.01 | [0.0, 0.0] | 0.5 | NO |
| `reasoning_depth.token_matched.3.graph_over_caption.estimate` | -0.8 | 0.01 | [-0.9, -0.6799999999999999] | 0.9998875397677801 | NO |
| `ood_contrasts.token_matched.graph_over_caption.estimate` | -0.11499999999999999 | 0.01 | [-0.13999999999999999, -0.08] | 0.9987664784245228 | NO |

## Robustness check

- Result complete: True
- Registry validated: True
- Model matches registration: True
- Dataset matches registration: True
- Model eligible for scientific decision: True
- Overall robustness pass: False

This Markdown file mirrors `decision_report.yaml`, which is the authoritative record.
Threshold changes require a new registration before rerunning.
