# Synthetic dataset protocol

## Generation

The authoritative implementation is `src/recoalign/synthetic_world/`; the root-level
`synthetic_world/` package is a compatibility import for the pre-existing Phase-1 runners.

Generate the 1,000-sample reference artifact with:

```powershell
recoalign generate-synthetic `
  --config configs/synthetic_benchmark.yaml `
  --count 1000 `
  --output outputs/synthetic_world_example
```

The generator creates `dataset.jsonl`, `scenes.json`, `metadata.jsonl`, `manifest.json`, and one
reconstructable bundle per sample:

```text
samples/<sample-id>/
  image.png
  scene.json
  graph.json
  metadata.json
```

Aggregate images are also indexed under `images/`. The JSONL file is portable benchmark data; image
paths in a materialized run bind the record to its generated artifact root.

## Required validation gates

Generation is complete only if all of the following pass:

1. JSON Schema validation against `schemas/synthetic_scene.schema.json`.
2. All object and relation values belong to the versioned ontology.
3. Relation endpoints resolve to declared objects.
4. The answer can be recomputed from the oracle graph or queried object attribute.
5. Supporting-edge count equals declared hop depth.
6. Caption and graph regenerate the same canonical fact hash.
7. Sample IDs are unique and the selected split policy reports no leakage.
8. Every materialized image exists and matches its SHA-256.
9. Re-rendering from world state reproduces identical PNG bytes.

The generated `validation_report.json` records these checks. Labels are not produced by an LLM and
must not be manually corrected. A generator bug is fixed in code and the affected version is
regenerated.

## Evaluation

Run the model-neutral evaluation interface with:

```powershell
recoalign evaluate-synthetic --config configs/synthetic_benchmark.yaml
```

It writes `metrics.json`, `predictions.jsonl`, and `decision_report.yaml`. Infrastructure runs require
at least three independent seeds. A configuration marked `evaluation.critical: true` requires at
least five. Accuracy is first computed per seed; aggregate output reports seed-level mean, sample
standard deviation, and a two-sided 95% Student-t confidence interval.

The shipped `reference` backend validates the pipeline only. Its decision report is always
`scientific_decision: NOT_APPLICABLE`; scientific GO/NO-GO decisions remain under the registered
experiment governance lifecycle.

## Distribution policy

Question family, hop depth, relation, split, renderer style, and ontology factors are configuration
variables. They must be selected before target-model evaluation. Do not rebalance or filter samples
after observing accuracy. Any protocol change increments the generator or manifest version.

## Artifact retention

Generated images and run outputs live under ignored `outputs/` paths. Track configuration, manifest,
schema, validation code, and aggregate generation reports in Git. This avoids committing thousands
of reproducible binaries while keeping every scientific decision auditable.
