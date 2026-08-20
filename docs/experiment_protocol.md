# Experiment protocol index after the research pivot

## Frozen historical experiments

| ID | Scientific question | Decision | Lifecycle |
| --- | --- | --- | --- |
| EXP001 | Does graph structure outperform fact-controlled text? | NO-GO | H001 falsified |
| EXP002 | Does performance require correct and complete relations? | GO | H002 supported in scope |
| EXP003 | Does graph advantage persist under OOD composition? | INCONCLUSIVE | H003 retired |
| EXP004 | Are semantics high while structured access is low? | NO-GO | H004 falsified |

The canonical historical contracts remain under `research/experiments/experiment_registry.yaml` and
`research/protocols/`. They must not be edited to fit the pivot narrative.

## Planned pivot experiments

| Planning ID | Candidate | Purpose | Registration state |
| --- | --- | --- | --- |
| PIVOT_EXP_A | PH001 | Semantic × relation × serialization factorial | not registered |
| PIVOT_EXP_B | PH002 | Relation-selective grounding probe | not registered |
| PIVOT_EXP_C | PH003 | Primitive versus composition decodability | not registered |
| PIVOT_EXP_D | PH004 | Cross-modal reasoning equivalence | not registered |

Planning IDs are deliberately absent from the governed registry until their protocols, configs,
dataset bindings, seeds, integrity assertions, and decision rules are complete.

## Common controls for future registration

- frozen backbone and revision;
- scene-disjoint splits;
- semantic and relation corruption manifests;
- fact and token equivalence;
- paired scene-level statistics and seed replication;
- explicit probe-capacity and random-label controls;
- retained failures and no post-result threshold changes.

## Evidence states

- `supported`: a bounded registered GO;
- `falsified`: a valid registered NO-GO;
- `retired`: a superseded claim retained without fabricating a decision;
- `inconclusive`: missing, failed-integrity, or incomplete evidence;
- `candidate`: a hypothesis that may not be reported as a result.
