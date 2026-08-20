# EXP004 Failure Taxonomy

Every incorrect Stage-3 response is retained under one registered category:

- `object_failure`: object or attribute identification errors;
- `relation_failure`: relation errors on non-multi-hop questions;
- `multi_hop_failure`: errors on questions requiring more than one licensed relation;
- `structure_misuse`: errors under corrupted structure or evidence inconsistent with the image.

Taxonomy counts are descriptive diagnostics. They do not alter accuracy, discard failures, or tune
prompts. The per-seed prediction rows remain the authoritative audit trail.
