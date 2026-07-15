# Results directory

This directory stores lightweight, reviewable experiment summaries. Large predictions, embeddings, checkpoints, and logs belong in ignored artifact storage.

Every reportable run should record at least:

```text
run_id, git_commit, config_path, dataset_version, model, pretrained,
seed, precision, checkpoint, status, started_at, completed_at,
metric_name, metric_value
```

Use explicit status values such as `pilot`, `partial`, `failed`, `complete`, and `reportable`. Only `reportable` runs should be copied into manuscript tables.
