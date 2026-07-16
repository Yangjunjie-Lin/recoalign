# Winoground experiment evidence template

No real result is committed by this repair unless a locally verified canonical run,
a cache-disabled verification run, and manual review evidence are available.

The local closure record must include the dataset revision, manifest SHA, annotation SHA, cached
run ID, no-cache run ID, git commit, config SHA, seed, metrics, maximum score difference, decision
differences, manual review count, reviewer, and promotion status. Keep the canonical cached run as
the only promotion candidate; the cache-disabled verification run must remain `complete`.

`result_template.json` contains no example metrics or run identifiers. Copy it only after the
pinned snapshot and both real runs have been verified. `reviewed_sample_ids_template.csv` contains
only its header and must be populated by an actual reviewer.

Reviewer notes alone cannot make a Winoground run reportable. Promotion requires the cache-enabled
canonical run, a separate complete cache-disabled verification run, an internally recomputed passing
comparison, schema-valid and recomputed predictions, all 400 mapping-review rows, and exact
annotation-to-inventory coverage. `promote-run` snapshots the comparison and review evidence into the
canonical run and records SHA-256 hashes; it does not accept a hand-authored comparison JSON. The
verification run remains `complete`, and this repository still contains no real reportable
Winoground result.
