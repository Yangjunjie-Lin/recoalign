# Research identity audit

## Audit scope

This audit covers the README, active documentation, research registries, experiment/config names,
and code-level naming visible to users. Historical archives, tests, and compatibility packages are
included as provenance checks but are not treated as active narrative unless they claim to define
the current method.

## Findings and disposition

| Location | Old or potentially ambiguous identity | Disposition |
|---|---|---|
| `docs/research_plan.md` | Retrieval-first phases, typed hard negatives, and region–phrase alignment | Rewritten around EXP001–EXP004, interface learning, OOD generalization, and preservation controls |
| `README.md` | Retrieval/OpenCLIP references could be read as the main contribution | Clarified as frozen capability-preservation controls and linked to the new identity documents |
| `docs/architecture.md` | Stable retrieval infrastructure described without an explicit active-line boundary | Updated to distinguish active interface research from compatibility/control infrastructure |
| `docs/baseline_protocol.md` and `docs/benchmark_evaluation_protocol.md` | Retrieval metrics and benchmark names | Retained; these are registered capability-preservation controls, not mechanism evidence |
| `src/recoalign/evaluation/retrieval.py`, `src/recoalign/benchmarks/`, baseline configs, and retrieval tests | Retrieval-oriented module/class names | Retained for reproducibility, compatibility, and controls; no active ReCoAlign method claim is attached |
| `archive/` and `docs/restructuring_report.md` | Earlier retrieval/alignment directions | Retained as history; not deleted or presented as the frozen contribution |
| `models/alignment/`, `models/structure_encoder/`, and `models/reasoning_interface/` | Transitional module names | Retained because they are API boundaries and changing them would break imports/history; active docs now define their role through the interface hypothesis |
| `research/experiments/` and `research/protocols/` | EXP001–EXP004 names | Already aligned with question-driven mechanism validation; indexed in `docs/experiment_protocol.md` |

## Identity decision

The active identity is now **ReCoAlign — Structured Reasoning Interface Learning for
Vision-Language Models**. Retrieval, legacy alignment, and older exploratory directions remain
auditable controls or history. No experiment result, failed direction, or benchmark artifact was
deleted or rewritten by this migration.

## Required follow-up

Future code and documentation changes must use the claim boundary in
[`docs/claim_boundary.md`](../docs/claim_boundary.md). Any new method or benchmark claim must be
registered before being described as evidence.
