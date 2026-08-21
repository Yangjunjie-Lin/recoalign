# PH001 Pivot Validation

This directory contains the preregistered decomposition and minimal behavioral tests authorized by
the research-pivot decision. It does not contain a model, adapter, loss, or benchmark expansion.

## Decomposition

- `H-A`: primitive semantic availability limitation;
- `H-B`: relational evidence integration and representation-format sensitivity.

## Experiments

- `A1`: direct image-only shape, color, and one-hop relation decisions;
- `A2`: image, object evidence, relation evidence, and their complete union;
- `A3`: identical relation facts in natural language, triples, and JSON.

The three experiments use the existing frozen synthetic scenes, five registered seeds, one pinned
LLaVA-1.5 checkpoint, deterministic decoding, paired statistics, and explicit integrity gates.
Pilot results use the first three seeds; final decisions require all five.

Raw resumable execution artifacts are written under `outputs/pivot_validation/`. Final lightweight
metrics, compressed predictions, provenance, and `pivot_decision.yaml` are promoted under
`research/pivot_validation/results/` only after the five-seed integrity gate passes.
