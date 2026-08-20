# Final repository audit

## Outcome

The research infrastructure is auditable, but the repository is not scientifically submission-ready because real-VLM and comprehensive matrix evidence is incomplete.

## Experiment artifacts

| Experiment | Role | Config | Manifest | Metrics | Predictions | Claim eligible |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| EXP001 | infrastructure_validation | True | True | 7 | 7 | False |
| EXP002 | infrastructure_validation | True | True | 6 | 6 | False |
| EXP003 | infrastructure_validation | True | True | 6 | 6 | False |
| EXP004 | infrastructure_validation | True | True | 6 | 5 | False |

## Code and release hygiene

- Tracked Python files: 251
- Duplicate-content groups requiring review: 0
- Temporary/validation output candidates retained: 12
- Tracked local-path findings: 0
- Potential secret findings: 0
- Dirty worktree: True

Ignored experiment outputs were not deleted: failed and incomplete runs are scientific audit evidence, and some may be user-owned artifacts.
