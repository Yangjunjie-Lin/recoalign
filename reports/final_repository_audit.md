# Final repository audit

## Outcome

The research infrastructure and frozen LLaVA evidence are auditable, but the repository is not scientifically submission-ready because the registered evidence decision is NO-GO.

## Experiment artifacts

| Experiment | Role | Execution | Decision | Config | Manifest | Metrics | Predictions | Claim eligible |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| EXP001 | scientific_evidence | complete | NO-GO | True | True | 7 | 7 | True |
| EXP002 | scientific_evidence | complete | GO | True | True | 6 | 6 | True |
| EXP003 | scientific_evidence | failed_integrity_gate | INCONCLUSIVE | True | True | 6 | 6 | False |
| EXP004 | scientific_evidence | complete | NO-GO | True | True | 6 | 5 | True |

## Code and release hygiene

- Tracked Python files: 253
- Duplicate-content groups requiring review: 0
- Temporary/validation output candidates retained: 12
- Tracked local-path findings: 0
- Potential secret findings: 0
- Dirty worktree: True

Ignored experiment outputs were not deleted: failed and incomplete runs are scientific audit evidence, and some may be user-owned artifacts.
