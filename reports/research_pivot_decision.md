# ReCoAlign Research Pivot Decision

## Final decision

- **NO-GO:** paper writing, scientific submission, and ReCoAlign model development.
- **GO:** a bounded hypothesis-revision stage beginning with preregistration of PIVOT_EXP_A.

## Rejected direction

The active claim that VLMs preserve sufficient visual semantics but lack a graph interface is
rejected under the frozen LLaVA-1.5 evidence. H001 and H004 are `falsified`; H003 is `retired`
because its evidence is inconclusive and its prerequisite graph advantage failed; H002 is
`supported` only within its registered scope.

## Selected direction

**PH001 — Serialization-Conditioned Semantic-Structural Integration Failure.**

Working hypothesis:

> Compositional failure results from the joint limits of visually grounded semantic availability
> and format-conditioned integration of correct relational evidence. Correct relations help when
> supplied, but graphs are not a privileged reasoning interface.

This hypothesis is provisional. It becomes scientifically active only after PIVOT_EXP_A receives a
complete preregistration and passes its falsification gates.

## Why this direction was selected

- It explains the observed combination `Caption > Graph`, `Correct > Corrupted Relations`, and low
  registered SAS without contradicting any frozen result.
- It does not assume that a linear probe measures all semantic information.
- It is testable using a minimal factorial intervention on the existing controlled world.
- It does not require a new architecture or trainable model.

## Readiness gates

| Gate | Current state |
| --- | --- |
| Old hypothesis rejected and retained | PASS |
| Claim-level evidence audit | PASS |
| Four alternative hypotheses specified | PASS |
| Minimal falsification experiment for each candidate | PASS |
| Primary direction selected | PASS |
| PIVOT_EXP_A preregistered | PENDING |
| PIVOT_EXP_A real-model evidence | PENDING |
| Independent backbone replication | PENDING |
| Paper writing | BLOCKED |

## Repository migration rule

Existing EXP001–EXP004 protocols, predictions, decisions, figures, and manifests remain immutable.
Narrative documents may mark the old identity as falsified and introduce the pivot candidate, but
no old outcome may be relabeled. New experiment IDs must not enter the governed experiment registry
until their protocol, config, dataset binding, seed policy, and decision rule are complete.

## Next authorized action

Prepare and review the PIVOT_EXP_A preregistration. No model implementation is authorized by this
decision.
