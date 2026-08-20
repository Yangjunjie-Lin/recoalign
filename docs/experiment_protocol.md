# Experiment protocol index

The frozen research identity is tested through registered, question-driven experiments. The
canonical contracts live in `research/experiments/experiment_registry.yaml` and
`research/protocols/`; this page is the human-readable index.

| ID | Scientific question | Protocol |
|---|---|---|
| EXP001 | Does correct graph structure improve reasoning beyond information-matched text? | [`research/protocols/EXP001_graph_vs_text.md`](../research/protocols/EXP001_graph_vs_text.md) |
| EXP002 | Does the effect require correct and complete structure rather than extra tokens or graph-like formatting? | [`research/protocols/EXP002_graph_ablation.md`](../research/protocols/EXP002_graph_ablation.md) |
| EXP003 | Does structured access support unseen composition generalization? | [`research/protocols/EXP003_ood_composition.md`](../research/protocols/EXP003_ood_composition.md) |
| EXP004 | Where does a real VLM fail: semantic availability, structured accessibility, or execution? | [`research/protocols/EXP004_interface_diagnosis.md`](../research/protocols/EXP004_interface_diagnosis.md) |
| TRAIN001–003 | Can a learned interface be trained, aligned, and transferred without oracle graph inference input? | [`research/experiments/training_registry.yaml`](../research/experiments/training_registry.yaml) |

## Common controls

Every claim-bearing run must record the registered split, seed set, prompt/evaluator version,
checkpoint or manifest identity, environment, predictions, metrics, and decision report. Random,
corrupted, length-matched, parameter-matched, and no-structure controls are retained where the
protocol requires them.

## Evidence states

- `verified`: claim-eligible evidence satisfies the registered gate;
- `infrastructure_only`: dry-run, fixture, toy, or ReferenceVLM validation;
- `blocked` or `pending`: required data, checkpoint, API, or result is absent;
- `failed`: an observed negative result, which remains part of the audit trail.

Retrieval benchmarks are invoked through the separate capability-preservation protocol and must
not be merged into the mechanism metrics.
