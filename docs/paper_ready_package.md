# Paper-ready research package

Phase 6 maps every proposed paper claim to registered experiments and retained artifacts. It keeps
three states distinct:

- `verified`: claim-eligible scientific evidence satisfies the frozen protocol;
- `infrastructure_only`: a toy, fixture, dry-run, or ReferenceVLM result validates implementation;
- `pending`: required claim-eligible evidence is absent or incomplete.

Build and validate the package with:

```bash
python -m recoalign build-paper-package
python -m recoalign validate-paper-package
```

The build writes the evidence map, candidate frozen registry, LaTeX exports, SVG figures,
reproducibility records, anonymous-submission documentation, repository audit, and integrity report.
Missing results are rendered as pending; the exporter never inserts placeholder scores.

## Freeze and tag policy

`experiments/frozen_registry.yaml` locks protocol identities and records current config, dataset,
checkpoint, seed, commit, and hash state. New results may be appended under the locked protocol, but
protocol mutation requires a new registry version.

The following command checks the paper tag gates:

```bash
python -m recoalign create-paper-tag
```

Use `--create` only after the scientific decision is `GO`. The command refuses to create
`v2.0-paper-ready` unless the worktree is clean, the frozen registry matches `HEAD`, and the final
scientific readiness report is `GO`.

## Current status

The paper-package implementation is complete, but the scientific submission decision is `NO-GO`.
The comprehensive real-VLM matrix and real-VLM mechanistic evidence are incomplete. Toy and
ReferenceVLM artifacts therefore remain non-claim-eligible.
