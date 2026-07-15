# Phase-0 architecture

ReCoAlign separates stable research infrastructure from hypothesis-specific code.

```text
configs/                    Committed experiment intent
manifests/                  Dataset, checkpoint, and sweep provenance
schemas/                    Runtime-enforced record contracts
src/recoalign/data/         Dataset-neutral records and manifest verification
src/recoalign/benchmarks/   Unified benchmark adapter contract
src/recoalign/models/       Encoder abstractions and OpenCLIP integration
src/recoalign/methods/      ReCoAlign method variants, added after diagnosis
src/recoalign/generation/   Hard-positive and hard-negative generation
src/recoalign/training/     Training loops, losses, optimization, checkpointing
src/recoalign/evaluation/   Metrics and prediction serialization
src/recoalign/experiments/  Run lifecycle, promotion gates, and provenance
src/recoalign/analysis/     Tables, statistics, visualizations, failure taxonomy
environments/               Versioned bootstrap profiles
results/                    Lightweight reviewed summaries only
paper/                      Manuscript-controlled figures and tables
```

## Dependency direction

Core configuration, schema validation, manifests, experiment records, and metrics remain importable
without PyTorch or OpenCLIP. Benchmark and model integrations may depend on optional ML packages.
This keeps CI fast and prevents ML environment failures from corrupting provenance records.

## Experiment lifecycle

1. Validate a committed YAML configuration.
2. Bind dataset/checkpoint manifests and capture their digests.
3. Create a run directory with resolved config, manifest snapshots, verification, and environment.
4. Execute evaluation or training and write predictions to ignored artifact storage.
5. Finalize with finite metrics and a non-reportable status.
6. Promote only a clean, verified, reviewed run to `reportable`.
7. Build manuscript tables from schema-valid reportable records.

## Research boundary

Phase 0 deliberately does not claim a ReCoAlign method. Modules under `methods`, `generation`, and
`training` remain minimal until baseline diagnostics justify a specific hypothesis.
