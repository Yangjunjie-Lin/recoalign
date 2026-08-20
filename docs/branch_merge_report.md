# Branch merge report

## Consolidation

| Branch | Result |
| --- | --- |
| `origin/main` | Fast-forwarded local `main` to `9acc520`. |
| `origin/develop` | Verified as the older development line; its effective history is contained in main's merged history. |
| `origin/experiment/winoground-first-real-closure` | Included through the test/concept ancestry; Winoground evidence is retained. |
| `origin/test/concept` | Included through fast-forward history. |
| local `test/concept` | Included through local commit `5b271e0` (`test4`), then retained as a historical ref. |

No merge commit or conflict resolution was required: all valid branch histories were fast-forward
compatible. The resulting canonical development branch is local `main`, currently at commit
`9fbef27` after the restructuring commit.

The local `develop`, Winoground experiment, and `test/concept` refs are deleted after their commits
were verified as ancestors of `main`; their commits remain reachable from `main` and the remote refs.
No remote push or remote branch deletion was performed, so GitHub's existing remote branches remain
unchanged until an explicit publishing step.

## Conflict handling

There were no textual merge conflicts. The main consolidation risk was generated artifacts rather
than competing source edits. The large Phase 2/3 image and feature files were removed from tracking
and copied to ignored `.local_artifacts/`; their lightweight manifests and reports remain archived.

## Tag

The local tag `v1.0-pre-structure-interface` marks the post-consolidation, pre-training research
platform. It is a local reproducibility marker until explicitly pushed.
