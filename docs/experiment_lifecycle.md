# Experiment lifecycle

```bash
recoalign validate-config configs/baseline/openclip_vit_b32.yaml

recoalign init-run \
  --config configs/baseline/openclip_vit_b32.yaml \
  --output-root outputs

# Run the benchmark/training command and produce a metrics JSON object.

recoalign finalize-run outputs/<run-id> \
  --metrics outputs/<run-id>/metrics.pending.json \
  --status complete

recoalign build-table \
  --results-root results \
  --status reportable \
  --metrics i2t_R@1 t2i_R@1 mean_recall
```

`outputs/` is ignored because predictions and checkpoints may be large. Small reviewed run records
may be copied under `results/` without raw embeddings or weights.
