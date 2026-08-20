# Governed run artifacts

`recoalign run-experiment EXPxxx` creates append-only bundles under
`runs/<experiment-id>/<run-id>/`. Generated bundles are intentionally ignored by Git; this README is
the only tracked file in the directory.

Every top-level bundle contains `config.resolved.yaml`, `command.txt`, `environment.txt`,
`git_commit.txt`, `seed.txt`, `metrics.json`, `log.txt`, and `manifest.json`. Completed empirical runs
also contain `decision_report.md` and one evidence directory per seed. See
`docs/reproducibility.md` for the complete contract.
