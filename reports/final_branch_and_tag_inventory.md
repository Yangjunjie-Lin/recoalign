# Final Branch and Tag Inventory

Inventory date: 2026-08-25

## Long-term refs

| Ref | Exact SHA | Status / recommendation |
| --- | --- | --- |
| `origin/main` | `5377eba6607575b47cb8a57b87537c7e748615c6` | Closeout and release content integrated by PR #15. |
| `origin/codex/claim-evidence-execution` | `a80882071a6cf17c275453319d78d879c1546e3a` | Frozen closeout source; retain until release verification. |
| `origin/release/recoalign-closeout` | `0bea0cf8ff6d815e549aade2323d3c0ca98428b1` | Temporary merged release branch. |
| `release/recoalign-closeout-finalize` | based on `5377eba6607575b47cb8a57b87537c7e748615c6` | Temporary metadata and release-closure branch. |
| `recoalign-evidence-freeze-2026-08-25` | tag object `3fd776cab8bfd50e1d211c863d8fa91a651a9de7`; commit `a80882071a6cf17c275453319d78d879c1546e3a` | Retain permanently. |
| `recoalign-negative-evidence-v1` | pending | Create only after validation; retain permanently. |

## Historical remote branches

| Branch | Recorded tip SHA | Post-release recommendation |
| --- | --- | --- |
| `origin/agent/publish-paper-results` | `e53c718365cf1fb8fe6716b77b242cf77103cc55` | Eligible for deletion after owner review. |
| `origin/develop` | `671a06e07309b667d8e324e7ae852a50281a4e4f` | Retain or delete only under owner branch policy. |
| `origin/experiment/winoground-first-real-closure` | `34df0199964c7b08ecfb72ba06daacf855c3a94e` | Eligible for deletion after owner review. |
| `origin/test/concept` | `d21fa517767bb75c2ab5b4e6afe740207560925a` | Eligible for deletion after owner review. |

No historical branch is deleted by this release operation. After both permanent tags and main are
verified, the temporary release branch may be deleted. The recommended durable ref set is `main`,
`recoalign-evidence-freeze-2026-08-25`, and `recoalign-negative-evidence-v1`.
