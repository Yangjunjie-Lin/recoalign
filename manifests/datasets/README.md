# Dataset manifests

Every dataset snapshot used by an experiment must have a committed YAML manifest that satisfies
`schemas/dataset_manifest.schema.json`.

A reportable manifest must declare the exact files required for the evaluated split, including size
and SHA-256 where practical. Template manifests with an empty `files` list may be used for planning
but cannot pass reportability gates.
