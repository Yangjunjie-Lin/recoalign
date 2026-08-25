# Release Branch Integration Report

## Topology before integration

- Frozen source branch: `origin/codex/claim-evidence-execution`
- Frozen source SHA: `a80882071a6cf17c275453319d78d879c1546e3a`
- Main before integration: `fe3b37f1cffc58329a4823ec3af20b3ade7ca6d3`
- Merge base: `4a69528d8cdddbf67c533cf36af62286200bf383`
- Main unique commits: 1
- Closeout unique commits: 12

Main's unique commit was:

```text
fe3b37f Merge pull request #14 from Yangjunjie-Lin/codex/claim-evidence-execution
```

## Integration operation

Branch `release/recoalign-closeout` was created directly from the frozen source commit. The command
equivalent to `git merge --no-ff origin/main` created merge commit
`3fa9ccd793ebbcaecae3bc283281fc1a44f1268f` with parents `a808820…` and `fe3b37f…`.

- Conflicts: none.
- Conflict resolutions: none required.
- Tree delta between the frozen commit and integration merge: none.
- Frozen evidence path delta: none.
- Rebase, squash, cherry-pick replacement, force push, or history rewrite: none.

## Main result

- Pull request: [#15](https://github.com/Yangjunjie-Lin/recoalign/pull/15), merged with a merge commit.
- Resulting main integration SHA: `5377eba6607575b47cb8a57b87537c7e748615c6`.
- Frozen evidence unchanged: yes; byte-level manifests are revalidated as part of release closure.

Finalization PR [#16](https://github.com/Yangjunjie-Lin/recoalign/pull/16) added deterministic
archives, clean-checkout tests, cross-platform frozen-byte rules, and Python 3.10/3.11 integrity
compatibility. It merged as `30050ae48c960b4eeca242fb43570d9305fd2162` after both CI matrix
jobs passed. These changes are release engineering and do not alter frozen scientific evidence.

## Privacy-only release-tree adjustment

The release security scan found local user absolute paths in three auxiliary PIVOT_EXP_A3 runtime
inventory files. None was included in the frozen v1 artifact manifest. The release tree replaces
those local-path-bearing copies with privacy-redacted provenance stubs that identify their original
Git blobs and the permanent evidence-freeze tag. No prediction, metric, decision, protocol,
manifest-bound artifact, or scientific interpretation changed. Git history was not rewritten.
