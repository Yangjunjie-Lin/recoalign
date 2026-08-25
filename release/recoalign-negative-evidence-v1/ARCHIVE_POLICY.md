# Archive Policy

The long-term preservation anchors are:

- branch `main` after closeout integration;
- annotated tag `recoalign-evidence-freeze-2026-08-25` at
  `a80882071a6cf17c275453319d78d879c1546e3a`;
- annotated tag `recoalign-negative-evidence-v1` at the validated release state; and
- deterministic archives plus their `SHA256SUMS` file.

Tags are immutable and must never be moved or overwritten. Frozen evidence ancestry must not be
rebased, squashed, cherry-picked as a replacement lineage, force-pushed, or otherwise rewritten.
Historical branches may be deleted only after their exact tip SHA is recorded in
`reports/final_branch_and_tag_inventory.md`. The temporary release branch may be deleted after its
merge commit and both permanent tags are verified.

An external archival deposition may be made later without changing scientific claims. No DOI or
deposition status may be asserted until the external service has actually issued it. After main
integration, validation, tag creation, and release-asset verification, GitHub repository archival
is recommended. Archival must not precede release completion.
