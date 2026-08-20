# ReCoAlign research identity — pivot version

## Project name

ReCoAlign

## Working research name

Semantic–Structural Integration Diagnostics for Vision-Language Models

## Identity decision

The former identity, Structured Reasoning Interface Learning, is no longer active. Frozen
LLaVA-1.5 evidence falsified graph superiority and the required high-semantic/low-structured-access
diagnosis. The original freeze is retained in `research/frozen_hypothesis.yaml` with status
`falsified` and documented in `research/rejected_hypothesis.md`.

## Core problem

Current VLM compositional errors cannot be explained as a graph-interface gap alone. The frozen
evidence shows three distinct observations:

1. fact-equivalent fluent captions outperform graph triples;
2. correct externally supplied relations outperform corrupted relations;
3. the registered visual semantic probe reports low availability.

The unresolved problem is how visually grounded semantics, relational correctness, and evidence
serialization jointly determine reasoning—not whether a graph is inherently the missing interface.

## Candidate hypothesis

The selected pivot candidate, PH001, is:

> On controlled compositional tasks, frozen VLM failure is produced by the joint limits of visually
> grounded semantic availability and serialization-conditioned integration of correct relational
> evidence. Correct relations help once supplied, but graph serialization is neither necessary nor
> privileged.

This is a candidate awaiting falsification, not a supported claim.

## Research questions

### RQ1 — Semantic availability and grounding

Which object, attribute, relation, binding, and compositional variables are recoverable from frozen
visual representations under scene-disjoint, capacity-controlled probes?

### RQ2 — Relational evidence use

Does correct relational evidence retain a causal effect after semantic correctness, fact coverage,
token budget, and serialization are controlled?

### RQ3 — Integration and serialization

Do semantic correctness and relation correctness interact, and is the interaction conditioned by
fluent-language versus canonical-triple serialization?

### RQ4 — Bottleneck discrimination

Can the observed failures be distinguished from relational grounding failure, a compositional
representation bottleneck, or cross-modal reasoning alignment failure?

## Evidence status

| Evidence | Decision | Pivot role |
| --- | --- | --- |
| EXP001 Graph vs Caption | NO-GO | Establishes caption-over-graph observation; mechanism unresolved |
| EXP002 Structural Necessity | GO | Supports sensitivity to correct supplied relations |
| EXP003 OOD | INCONCLUSIVE | Retained integrity failure; not promoted |
| EXP004 Diagnosis | NO-GO | Rejects the high-SAS/low-StAS interface pattern |
| PIVOT_EXP_A | Not registered | Required minimal factorial falsification |

## Scope and non-goals

- No graph-interface claim is active.
- Low SAS is not treated as proof that all semantic information is absent.
- EXP002 is not evidence of an internal scene graph.
- No ReCoAlign model, adapter, or loss is authorized during hypothesis revision.
- Retrieval remains a capability-preservation control, not the scientific contribution.
- Existing predictions, metrics, protocols, and failed runs remain immutable.
- Paper writing remains blocked.

## Sources of truth

- Old falsified freeze: `research/frozen_hypothesis.yaml`
- Pivot candidate: `research/pivot_hypothesis.yaml`
- Candidate registry and scores: `research/hypotheses/pivot_candidate_registry.yaml`
- Claim audit: `research/evidence_audit.md`
- Minimal experiments: `research/next_experiment_plan.md`
- Pivot decision: `reports/research_pivot_decision.md`
