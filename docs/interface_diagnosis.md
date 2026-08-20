# EXP004 Structured Interface Diagnosis

EXP004 separates three claims that are often conflated in VLM analysis:

1. Visual semantic availability (SAS): can frozen visual representations support object,
   attribute, relation, and composition probes?
2. Structured accessibility (StAS): can the model expose or preserve relational structure through
   graph reconstruction, latent relation probes, or semantic-equivalence consistency?
3. Reasoning execution (RES): does the model answer correctly with image-only, oracle graph, and
   corrupted graph inputs?

The diagnosis runner records unavailable representation APIs explicitly. It never substitutes an
oracle feature for a model representation. ReferenceVLM is an infrastructure sanity baseline and
cannot produce claim-bearing evidence. A real-model diagnosis requires a pinned checkpoint, the
registered seeds, and a complete reviewed output bundle.

Run the registered reference validation with:

```bash
recoalign run-interface-diagnosis --model reference
```

Use `--dry-run` to validate the EXP004 bindings without inference. `run-experiment --experiment
EXP004` is an equivalent governed entry point. Outputs contain `metrics.json`, `analysis.json`,
`decision_report.yaml`, per-seed datasets/metrics/predictions, and three diagnostic figures.
