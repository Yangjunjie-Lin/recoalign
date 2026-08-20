# Multi-VLM reporting protocol

Reports are generated from the full matrix plan rather than from discovered result
files alone. This prevents missing and failed experiments from disappearing.

The reporting system creates:

- Markdown and LaTeX main, ablation, and generalization tables;
- multi-seed and paired statistics;
- capability-preservation and failure-taxonomy analyses;
- five SVG figure slots whose captions disclose pending evidence;
- benchmark, ablation, mechanism-consistency, and GO/NO-GO reports.

Fixture, dry-run, ReferenceVLM, and injected test-backend results always set
`scientific_evidence_eligible: false`. The comprehensive decision remains
`INCONCLUSIVE` until multiple real model families, multiple datasets, sufficient
seeds, capability preservation, and mechanism ablations are complete.
