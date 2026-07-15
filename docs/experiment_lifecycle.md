# Experiment lifecycle

1. Validate a committed configuration with `recoalign validate-config`.
2. Initialize a run. The command validates and snapshots dataset/checkpoint manifests, verifies
   declared files, captures environment metadata, and writes a schema-valid `run.json`.
3. Execute evaluation or training. Large predictions, caches, and checkpoints remain in ignored
   artifact storage.
4. Finalize the run as `pilot`, `partial`, `failed`, or `complete` with finite schema-valid metrics.
5. Review a `complete` run and use `promote-run` to apply the reportability gates.
6. Generate manuscript tables only from schema-valid `reportable` runs.

A run is never promoted merely because its metric is favorable. Failed and partial runs remain
available for diagnosis and must not be silently overwritten.
