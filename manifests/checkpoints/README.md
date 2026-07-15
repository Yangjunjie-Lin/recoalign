# Checkpoint manifests

Every pretrained identifier or local checkpoint used by an experiment must have a committed YAML
manifest satisfying `schemas/checkpoint_manifest.schema.json`.

Remote registry identifiers may have no local files. Fixed local checkpoints should record file
paths, byte sizes, SHA-256 hashes, source, and applicable license terms.
