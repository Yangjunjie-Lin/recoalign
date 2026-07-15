# Dataset manifests

Every dataset snapshot used by an experiment must have a committed YAML manifest that satisfies
`schemas/dataset_manifest.schema.json`.

Generate the real Flickr30K, MS COCO, and SugarCrepe manifests with the preparation commands in
`docs/data_preparation.md`. The `*.example.yaml` files are planning templates only and cannot pass
reportability gates because they do not declare verified files.

For image benchmarks, pilot runs may use a size-only inventory. A run can be promoted to
`reportable` only when the prepared manifest records `processing.image_hashes: true` and every
referenced test image passes SHA-256 verification.
