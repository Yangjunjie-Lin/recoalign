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

Canonical and verification runs are audited against the same config, environment, manifests,
normalized annotation, predictions, decisions, and all recomputable metrics. Prediction sample ID,
category, tags, and row order must match the normalized annotation exactly. Table collection
revalidates the promotion hashes, comparison gates, 400-row review, artifact digests, and the unique
retained complete verification run.

## Local human-review workflow

The committed `reviewed_sample_ids.csv` is a queue until a person has inspected every row. It is
not reviewed evidence while the human fields are blank. Start the local-only helper from the
repository root:

```bash
python scripts/review_winoground.py \
  --run-dir outputs/openclip_vit_b32_laion2b_winoground_zero_shot/wg-openclip-b32-laion2b-canonical-20260717-13d2c51 \
  --review-csv reports/experiments/winoground/reviewed_sample_ids.csv
```

The helper binds only to a loopback address, uses no external assets or AI API, and never uploads
images. It shows both images, both captions, the canonical score matrix, sample ID, tags, and the
machine decision group. No review value is preselected. Each completed row is atomically saved, can
be resumed after interruption, and is then read-only in the helper.

For every row, the reviewer must check that image 0 corresponds to caption 0, image 1 corresponds
to caption 1, both images are readable, neither caption is shifted, and the annotation has no
obvious defect. Set `mapping_checked=true`, select `pass`, `issue`, or `uncertain`, and select
`none`, `possible`, or `confirmed` for the annotation issue. `issue` and `uncertain` rows require
non-empty notes. A `pass` row does not require notes.

Validate current coverage without opening the page:

```bash
python scripts/review_winoground.py \
  --run-dir outputs/openclip_vit_b32_laion2b_winoground_zero_shot/wg-openclip-b32-laion2b-canonical-20260717-13d2c51 \
  --review-csv reports/experiments/winoground/reviewed_sample_ids.csv \
  --check-only
```
