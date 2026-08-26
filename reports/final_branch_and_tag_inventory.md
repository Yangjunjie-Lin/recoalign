# Final Branch and Tag Inventory

Inventory date: 2026-08-26

## Long-term refs

| Ref | Exact SHA | Status / recommendation |
| --- | --- | --- |
| `origin/main` | `33b85e64800a20028be7829dbe833c2f4af83017` | Release-status integration completed by merge-commit PR #17; release tag target. |
| `origin/codex/claim-evidence-execution` | `a80882071a6cf17c275453319d78d879c1546e3a` | Frozen closeout source; exact tip is permanently anchored by the evidence-freeze tag. |
| `origin/release/recoalign-closeout` | `0bea0cf8ff6d815e549aade2323d3c0ca98428b1` | Temporary merged release branch; eligible for deletion after owner review. |
| `origin/release/recoalign-closeout-finalize` | `af3d5dc7e49b232bbf93077d58af4ab3b1663f39` | Temporary merged finalization branch; eligible for deletion after owner review. |
| `origin/release/recoalign-closeout-status` | `765e4e61904b2cfdbff8c66987182ec39d38761f` | Temporary merged status branch; eligible for deletion after owner review. |
| `release/recoalign-closeout-complete` | based on `33b85e64800a20028be7829dbe833c2f4af83017` | Post-release audit metadata branch; delete after merge and verification. |
| `recoalign-evidence-freeze-2026-08-25` | tag object `3fd776cab8bfd50e1d211c863d8fa91a651a9de7`; commit `a80882071a6cf17c275453319d78d879c1546e3a` | Retain permanently. |
| `recoalign-negative-evidence-v1` | tag object `6dbb62d97a940c726cf1ed62bcf9af2947ed321a`; commit `33b85e64800a20028be7829dbe833c2f4af83017` | Published release tag; retain permanently. |

## Historical remote branches

| Branch | Recorded tip SHA | Post-release recommendation |
| --- | --- | --- |
| `origin/agent/publish-paper-results` | `e53c718365cf1fb8fe6716b77b242cf77103cc55` | Eligible for deletion after owner review. |
| `origin/develop` | `671a06e07309b667d8e324e7ae852a50281a4e4f` | Retain or delete only under owner branch policy. |
| `origin/experiment/winoground-first-real-closure` | `34df0199964c7b08ecfb72ba06daacf855c3a94e` | Eligible for deletion after owner review. |
| `origin/test/concept` | `d21fa517767bb75c2ab5b4e6afe740207560925a` | Eligible for deletion after owner review. |

No historical branch was deleted by this release operation. All branch-tip SHAs above were recorded
before any deletion recommendation. The temporary `release/*` branches may be deleted after owner
review; `origin/codex/claim-evidence-execution` is also eligible after confirming the permanent
evidence-freeze tag remains available. The recommended durable ref set is `main`,
`recoalign-evidence-freeze-2026-08-25`, and `recoalign-negative-evidence-v1`.
