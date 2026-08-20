# Scientific experiment governance

## Scope and audit outcome

This framework governs evidence for the ReCoAlign claim that a structured reasoning interface can
bridge visual semantic representation and reasoning execution. It expressly does not authorize a new
model, neural module, loss, checkpoint, or training run.

The pre-governance audit found reusable YAML configurations, three Phase-1 synthetic evaluators, a
root `recoalign` CLI, JSON metrics, and partial Git/checkpoint provenance. It also found four gaps:

1. experiments were not linked to explicit falsifiable hypotheses;
2. three experiment-specific launch paths duplicated orchestration responsibility;
3. structured runs did not emit one complete, uniform provenance bundle;
4. statistical thresholds and GO/NO-GO rules were not machine-executable preregistrations.

The governance implementation closes these gaps with linked registries, a single registered runner,
schema-validated results, multi-seed statistics, and an automatic decision report.

## Authoritative records

- `research/hypotheses/hypothesis_registry.yaml`: scientific claims, predictions, and falsifiers.
- `research/experiments/experiment_registry.yaml`: hypothesis-linked experiments and decision rules.
- `research/protocols/`: the mandatory template and frozen Phase-1 protocols.
- `schemas/decision_report.schema.json`: GO / NO-GO / INCONCLUSIVE report contract.
- `results/schema/experiment_result.schema.json`: uniform governed-result contract.
- `runs/`: ignored execution artifacts; each run is self-contained and auditable.

An experiment without a valid `hypothesis_id` cannot pass `recoalign validate-research` and cannot be
launched by `recoalign run-experiment`. Benchmark improvement is not a valid registration objective;
the objective must state the scientific question and falsification logic.

## Canonical lifecycle

```text
hypothesis registration
        ↓
experiment + decision-rule registration
        ↓
protocol and YAML configuration validation
        ↓
single `recoalign run-experiment` entry
        ↓
per-seed evaluation + immutable provenance
        ↓
cross-seed statistics
        ↓
automatic schema-valid GO / NO-GO / INCONCLUSIVE decision_report.yaml (+ Markdown mirror)
```

Use the following commands from the repository root:

```bash
recoalign validate-research
recoalign list-experiments
recoalign run-experiment EXP001 --dry-run
recoalign run-experiment EXP001
recoalign decide-experiment runs/EXP001/<run-id>/metrics.json
```

`--dry-run` validates all bindings and writes provenance/manifest files, but does not generate a
dataset, load a model, or execute evaluation. `--seed` is an explicit diagnostic override; a result
with fewer than the registered minimum seeds cannot receive GO.

The committed `reference` backend is explicitly registered as `infrastructure_validation` and cannot
produce a scientific GO/NO-GO decision. It may exercise the complete pipeline, but the decision is
`INCONCLUSIVE` until a real frozen backend/checkpoint is preregistered with
`scientific_decision_allowed: true`.

Registry `conditions` name the scientific intervention (for example, seen versus unseen
composition). `input_conditions` bind those interventions to executable interface labels in the
configuration. Both are preregistered, so scientific variables cannot be silently replaced by runtime
labels.

## Change control

Protocol, primary metric, threshold, dataset version, seed count, or condition changes require a
committed registry/config revision before execution. Existing results retain registry and resolved
config SHA-256 digests. Post-hoc analyses must be labeled exploratory and cannot reverse the
preregistered decision.
