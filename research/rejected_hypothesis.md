# Rejected Hypothesis Record

## Rejected statement

> Vision-language models preserve sufficient task-relevant visual semantics but lack a graph-like
> structured intermediate interface for compositional reasoning.

## Decision

**REJECTED as the active ReCoAlign hypothesis on 2026-08-21.**

The decision applies to the frozen `structured_reasoning_interface_gap_v1` claim and its registered
LLaVA-1.5-7B evidence scope. It does not assert that no model can ever exhibit an interface gap; it
forbids ReCoAlign from presenting that mechanism as supported by the current evidence.

## Rejection basis

1. EXP001 falsified graph superiority under natural, fact-controlled, and token-matched comparisons.
2. EXP004 falsified the required high-SAS/low-StAS pattern and selected semantic failure.
3. EXP003 supplied no promotable OOD evidence.
4. EXP002 showed sensitivity to relation correctness, but that result does not rescue graph
   optimality or semantic preservation.

## Prohibited uses after rejection

- Do not describe graph structure as the missing VLM interface.
- Do not state that sufficient visual semantics are preserved unless a new registered probe battery
  establishes that premise.
- Do not reinterpret EXP002 as proof of internal graph representations.
- Do not repair EXP003 solely to recover the old narrative.
- Do not begin ReCoAlign model design or paper writing from the rejected premise.

## Retention rule

The hypothesis, protocols, negative results, and paper-evidence package remain in the repository.
Rejection changes lifecycle status; it does not delete or rewrite scientific history.
