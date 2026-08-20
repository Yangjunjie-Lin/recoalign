# Repository Migration Plan for the Research Pivot

## Migration objective

Move the active narrative from a falsified graph-interface claim to a provisional
semantic–structural integration diagnosis while preserving every frozen result and avoiding a
premature experiment or method implementation.

## Completed narrative migration

| Asset | Previous role | Pivot role |
| --- | --- | --- |
| `README.md` | Active graph-interface hypothesis | Pivot status, rejected premise, selected candidate |
| `research/frozen_hypothesis.yaml` | Active frozen hypothesis | Falsified historical freeze |
| `research/hypotheses/hypothesis_registry.yaml` | All H001–H004 testing | H001/H004 falsified, H002 supported, H003 retired |
| `docs/research_identity.md` | Structured interface identity | Semantic–structural diagnostic identity |
| `docs/research_plan.md` | Interface then model roadmap | Construct validity then minimal falsification |
| `docs/claim_boundary.md` | Conditions for interface claims | Explicit prohibition of unsupported claims |
| `docs/architecture.md` | Proposed interface architecture | Frozen diagnostic architecture; method deferred |
| `paper/README.md` | Manuscript workspace | Explicit paper-writing block |

## New pivot assets

- `research/evidence_audit.md`
- `research/rejected_hypothesis.md`
- `research/hypotheses/pivot_candidate_registry.yaml`
- `research/pivot_hypothesis.yaml`
- `research/next_experiment_plan.md`
- `reports/research_pivot_decision.md`

## Immutable assets

The following remain unchanged by the pivot and must keep their original hashes where registered:

- EXP001–EXP004 protocols and configs;
- synthetic benchmark generator and frozen dataset manifest;
- LLaVA checkpoint/model manifests;
- raw and compressed predictions;
- metrics, figures, analysis files, and decision reports;
- failed EXP003 first-seed bundle and integrity failure;
- `reports/paper_evidence/artifact_manifest.yaml`.

## Deferred changes

- Do not add PH001/PIVOT_EXP_A to the governed registries until its full contract exists.
- Do not rename experiment directories or CLI commands; they are reproducibility interfaces.
- Do not delete structure/model code; mark it historical until a future diagnosis justifies reuse or
  removal.
- Do not create a method branch, training checkpoint, paper section, or submission tag.

## Next migration gate

After PIVOT_EXP_A preregistration is reviewed, add a new hypothesis and experiment pair with new IDs,
protocol, config, seed policy, dataset binding, corruption manifest, power analysis, and decision
rule. `recoalign validate-research` must pass before any real-model inference.

## Rollback rule

If PH001 receives NO-GO, preserve it with a lifecycle transition and select among PH002–PH004 using
the prespecified discrimination plan. Do not restore the old Structured Reasoning Interface Gap.
