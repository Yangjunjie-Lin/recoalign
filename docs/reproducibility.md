# Reproducibility contract

Every governed structured experiment is traceable to code version, resolved configuration, dataset
version/generator manifest, frozen model/checkpoint identity, command, environment, and random seeds.
The canonical bundle is:

```text
runs/<experiment-id>/<run-id>/
├── config.resolved.yaml
├── command.txt
├── environment.txt
├── git_commit.txt
├── seed.txt
├── metrics.json
├── log.txt
├── manifest.json
├── decision_report.yaml        # authoritative GO / NO-GO / INCONCLUSIVE decision
├── decision_report.md          # readable mirror of the YAML decision
└── seeds/<seed>/               # per-seed data, predictions, raw metrics, and run manifest
```

`manifest.json` binds hypothesis/experiment IDs, protocol, dataset name/version/manifest, model
state, all seeds, Git state, resolved config digest, and both registry digests. `metrics.json` follows
`results/schema/experiment_result.schema.json`; per-seed raw metrics remain available rather than
being replaced by only an aggregate. `environment.txt` is JSON-formatted text containing the exact
Python/platform/package/accelerator inventory and Git metadata.

Generated datasets are reconstructed from generator version, resolved generation parameters, and
seed. Each seed directory contains its generated dataset manifest. EXP003 additionally records and
enforces `composition_overlap: false` for every seed.

Dry-runs produce the same top-level provenance bundle and an INCONCLUSIVE decision report, but
deliberately do not claim an empirical result. Run directories are append-only by convention and a
duplicate run ID is rejected.

A result is eligible for a paper table only when all of the following are known and machine
validated:

- repository commit, Git dirty state, and resolved configuration hash;
- dataset manifest snapshot, manifest hash, exact split, and successful file verification;
- checkpoint manifest snapshot and manifest hash;
- random seed, precision, package inventory, CUDA runtime, driver, and GPU model;
- run status, timestamps, metrics schema, and review metadata;
- whether a comparison is locally reproduced, evaluated from an official checkpoint, or quoted.

## Status semantics

- `pilot`: exploratory and not used for claims;
- `partial`: execution completed only in part;
- `failed`: invalid or interrupted run retained for diagnosis;
- `complete`: technically completed but not yet reviewed;
- `reportable`: promoted from `complete` after provenance and review gates pass.

`finalize-run` cannot create a `reportable` run. Promotion requires `promote-run`, which rejects
unknown or dirty Git states, missing dataset verification, invalid schemas, inconsistent commits,
and absent review identity.

## Manifest binding

`init-run` loads and validates both manifests referenced by the committed configuration, records
their SHA-256 digests, snapshots them into the run directory, and verifies declared local files.
An empty dataset file list is acceptable for a template but cannot be promoted to reportable.

## Environment capture

Every run records the Git branch, commit, dirty state, diff digest, untracked-file count, selected
package versions, full `pip freeze`, its SHA-256 digest, conda metadata, CUDA/cuDNN, NVIDIA driver,
and visible devices. A bootstrap environment file does not replace the captured resolved state.

## Generated data

Every hard positive or hard negative must preserve source sample ID, original caption,
transformation type, generator/version, filtering decisions, and audit status. Generated examples
must never be mixed into benchmark test sets used for final reporting.

## Structured-reasoning runs

Legacy direct structured-run records use the same reproducibility principles with a synthetic-world
manifest. New paper evidence must use the governed `run-experiment` lifecycle above:

- YAML configuration and integer seed are copied into the run context;
- generated `metadata.jsonl` and `scenes.json` identify every object, typed relation, query, answer,
  and composition split;
- `predictions.jsonl` contains one row per scene/condition pair;
- `metrics.json` reports condition accuracy and paired contrasts;
- `run.json` records model backend, checkpoint configuration, Git commit, config digest, and row
  counts;
- image files and feature arrays are generated artifacts and must remain outside Git.

The deterministic `ReferenceVLM` is a CI backend only. A real LLaVA-1.5 run must provide an injected
backend plus a locally verified checkpoint revision and quantization record; it must not silently
fall back to the reference backend.
