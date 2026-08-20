# Comprehensive multi-VLM evaluation report

Decision: **INCONCLUSIVE**

This report is generated from the complete registered matrix. Missing, blocked, dry-run, and failed cells remain visible in the tables; no selective result filtering is applied.

- Planned cells: 432
- Observed cell reports: 0
- Complete inference cells: 0
- Scientific blockers: fewer than two models have complete ReCoAlign evidence; fewer than three benchmarks have complete ReCoAlign evidence; paired ReCoAlign improvements are absent or inconsistent

## Evidence scope

The repository currently provides BaseVLM adapters and dry-run validation for LLaVA-1.5, LLaVA-NeXT, Qwen-VL, and InternVL. A ReCoAlign result is claim-eligible only when a trained interface checkpoint and backend hidden-context injection are both present. Dry-run and injected fixture results are infrastructure evidence, not paper claims.

## Artifacts

- `tables/table_main_results.md` and LaTeX counterpart
- `tables/table_ablation.md` and LaTeX counterpart
- `tables/table_generalization.md` and LaTeX counterpart
- `figures/figure_1_framework.svg` through `figure_5_failure.svg`
- `statistics/multi_seed_statistics.json`
