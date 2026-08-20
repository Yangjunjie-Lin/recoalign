# Research identity migration report

## Outcome

The repository narrative is frozen around **Structured Reasoning Interface Learning for
Vision-Language Models**. The active research question is whether a missing structured interface
between visual semantics and language reasoning explains compositional VLM failures and whether a
learned interface improves OOD composition generalization.

## Delivered artifacts

- [`docs/research_identity.md`](../docs/research_identity.md) — project identity, problem, RQs, scope, and evidence program.
- [`docs/contribution_framework.md`](../docs/contribution_framework.md) — four bounded contributions and evidence requirements.
- [`research/frozen_hypothesis.yaml`](../research/frozen_hypothesis.yaml) — machine-readable hypothesis freeze.
- [`docs/frozen_hypothesis.md`](../docs/frozen_hypothesis.md) — human-readable hypothesis and falsification gates.
- [`docs/claim_boundary.md`](../docs/claim_boundary.md) — permitted and prohibited claims.
- [`docs/experiment_protocol.md`](../docs/experiment_protocol.md) — question-driven experiment index.
- [`reports/research_identity_audit.md`](research_identity_audit.md) — old identity inventory and disposition.

## Narrative changes

- Replaced the old retrieval-first research plan with the EXP001–EXP004 mechanism-validation path.
- Clarified README and architecture boundaries: retrieval/OpenCLIP remains a capability-preservation
  control, not the active scientific contribution.
- Preserved retrieval modules, baseline configs, tests, and archive material for reproducibility and
  historical traceability.
- Did not modify benchmark results, checkpoints, split definitions, or scientific decisions.

## Scientific status

Identity migration is complete, but it is not scientific validation. Dry-runs, toy training, and
ReferenceVLM outputs remain infrastructure-only. The claim-eligible real-VLM and method evidence
gate remains pending, and the current scientific submission decision is `NO-GO`.
