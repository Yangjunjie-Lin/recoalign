# Reproducibility contract

A result is eligible for a paper table only when all of the following are known:

- repository commit and resolved configuration hash;
- dataset manifest and exact split;
- pretrained/checkpoint identifier and hash when locally stored;
- random seed, precision, library versions, CUDA runtime, and GPU model;
- run status, start/completion timestamps, and metric file;
- whether the value is locally reproduced, taken from an official checkpoint, or quoted from a paper.

## Status semantics

- `pilot`: exploratory and not used for claims;
- `partial`: execution completed only in part;
- `failed`: invalid or interrupted run retained for diagnosis;
- `complete`: technically completed but not yet reviewed;
- `reportable`: independently checked and permitted in manuscript tables.

Never mark a run `reportable` merely because its metric is favorable.

## Generated data

Every hard positive or hard negative must preserve source sample ID, original caption,
transformation type, generator/version, filtering decisions, and audit status. Generated examples
must never be mixed into benchmark test sets used for final reporting.
