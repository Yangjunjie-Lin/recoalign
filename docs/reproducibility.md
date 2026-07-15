# Reproducibility contract

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
