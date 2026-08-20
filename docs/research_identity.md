# ReCoAlign research identity

## Project name

ReCoAlign

## Full research name

Structured Reasoning Interface Learning for Vision-Language Models

## Core problem

Modern vision-language models can preserve useful visual semantic information while
still failing compositional questions. The failure may occur when visual semantics
must be made available as a structured intermediate representation for language-side
reasoning.

This project therefore studies the interface between visual semantics and reasoning,
not only the final benchmark score and not only image-text retrieval.

## Core hypothesis

The primary bottleneck is often insufficient structured reasoning accessibility rather
than complete absence of visual semantics:

```text
visual semantic availability
        ↓
structured representation accessibility
        ↓
reasoning execution
```

The hypothesis is deliberately falsifiable. A graph or structure is not assumed to be
useful, sufficient, or causally used; each claim requires a registered intervention,
control, and decision rule.

## Research questions

### RQ1 — Visual semantic availability

Do VLM representations preserve object, attribute, relation, and compositional
information needed by the task?

### RQ2 — Structured accessibility

Do VLMs expose that information to the reasoning path as a usable structured
intermediate representation?

### RQ3 — Learnable interface

Can a learned visual-to-structure interface improve compositional reasoning and
composition-disjoint generalization without relying on oracle structure at inference?

## Evidence program

| Question | Registered evidence |
|---|---|
| Graph structure versus information-matched text | EXP001 |
| Structural necessity and corruption controls | EXP002 |
| Unseen composition generalization | EXP003 |
| Semantic availability, structured accessibility, and execution diagnosis | EXP004 |
| Learned interface training and ablations | TRAIN001–TRAIN003 and the registered ablation matrix |

EXP001–EXP004 are mechanism and diagnosis protocols. They do not become claim-eligible
scientific evidence merely because a dry-run or ReferenceVLM run exists. The current
repository explicitly keeps the real-VLM evidence gate pending.

## Scope and non-goals

- The active scientific contribution is a structured reasoning interface for VLMs.
- Retrieval benchmarks, OpenCLIP adapters, and legacy alignment code remain available as
  capability-preservation controls and historical compatibility boundaries.
- No retrieval metric is silently relabeled as reasoning evidence.
- No oracle graph is an inference-time substitute for a learned interface.
- No result is changed, deleted, or promoted by this identity migration.

## Freeze status

The machine-readable hypothesis freeze is [`research/frozen_hypothesis.yaml`](../research/frozen_hypothesis.yaml).
Changes to the main hypothesis or claim boundary require a new research version and an
explicit registry decision; ordinary implementation work must not silently rewrite them.
