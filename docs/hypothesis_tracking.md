# Hypothesis tracking

The governed registry retains the original hypotheses because lifecycle transitions must not erase
failed research. Pivot candidates remain in a separate candidate registry until complete protocols
and configs exist; adding them prematurely would create orphan scientific registrations.

## Frozen hypothesis lifecycle

| ID | Hypothesis | Evidence | Lifecycle status | Interpretation |
| --- | --- | --- | --- | --- |
| H001 | Structured Reasoning Interface Gap | EXP001 NO-GO | **falsified** | Graph does not outperform controlled captions |
| H002 | Structural Necessity | EXP002 GO | **supported** | Correct supplied relations matter in frozen scope |
| H003 | Compositional Generalization | EXP003 INCONCLUSIVE | **retired** | Integrity gate failed and prerequisite graph advantage is absent |
| H004 | Three-Stage Interface Diagnosis | EXP004 NO-GO | **falsified** | Required high-SAS/low-StAS pattern is absent |

`supported` is bounded evidence, not universal proof. `falsified` records a valid NO-GO. `retired`
records a claim no longer worth executing without pretending its incomplete evidence was negative.

## Pivot candidates

| ID | Candidate | Score | State |
| --- | --- | ---: | --- |
| PH001 | Serialization-Conditioned Semantic-Structural Integration Failure | 18/20 | selected, requires falsification |
| PH002 | Relational Grounding Failure | 14/20 | alternative |
| PH003 | Compositional Representation Bottleneck | 13/20 | alternative |
| PH004 | Cross-Modal Reasoning Alignment Failure | 14/20 | alternative |

The machine-readable candidate registry is
`research/hypotheses/pivot_candidate_registry.yaml`. PH001 must not be added to the governed
`hypothesis_registry.yaml` until PIVOT_EXP_A has a complete bidirectionally linked experiment
registration, protocol, config, dataset binding, seed policy, and decision rule.

Run `recoalign validate-research` after any governed registry edit. Existing H001–H004 and
EXP001–EXP004 links remain valid despite lifecycle changes.
