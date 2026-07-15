# Phase-0 architecture

ReCoAlign separates stable research infrastructure from hypothesis-specific code.

```text
configs/                    Committed experiment intent
manifests/                  Dataset, checkpoint, and sweep provenance
src/recoalign/data/         Dataset-neutral records and manifest verification
src/recoalign/benchmarks/   Unified benchmark adapter contract
src/recoalign/models/       Encoder abstractions and OpenCLIP integration
src/recoalign/methods/      ReCoAlign method variants, added after diagnosis
src/recoalign/generation/   Hard-positive and hard-negative generation
src/recoalign/training/     Training loops, losses, optimization, checkpointing
src/recoalign/evaluation/   Metrics and prediction serialization
src/recoalign/experiments/  Run lifecycle and provenance
src/recoalign/analysis/     Tables, statistics, visualizations, failure taxonomy
schemas/                    Machine-readable record contracts
results/                    Lightweight reportable summaries only
paper/                      Manuscript-controlled figures and tables
```

## Dependency direction

Core configuration, manifests, experiment records, and metrics must remain importable without
PyTorch or OpenCLIP. Benchmark and model integrations may depend on optional ML packages. This
keeps CI fast and prevents environment failures from corrupting the provenance layer.

## Experiment lifecycle

1. Validate a committed YAML configuration.
2. Create a run directory and capture resolved config plus environment metadata.
3. Execute evaluation or training and write predictions to ignored artifact storage.
4. Finalize the run with finite numeric metrics and an explicit status.
5. Promote only reviewed `reportable` runs into manuscript tables.

## Research boundary

Phase 0 deliberately does not claim a ReCoAlign method. Modules under `methods`, `generation`, and
`training` remain minimal until baseline diagnostics justify a specific hypothesis.
