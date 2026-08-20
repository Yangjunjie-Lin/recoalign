# Repository restructuring report

## Removed from the active line

- Phase 0A projection/semantic-preservation diagnosis moved to `archive/semantic_failure/`.
- Phase 0B/0C semantic utilization/access diagnosis moved to `archive/alignment_failure/`.
- Phase 1 evidence-selection diagnosis moved to `archive/alignment_failure/`.
- Phase 2 semantic-binding diagnosis moved to `archive/alignment_failure/`.
- Original Phase 3 implementation moved to `archive/structure_validation/phase3_original/` after
  extracting reusable generator/graph/evaluation ideas into the new packages.
- Generated images, `.npy` feature arrays, local environments, and caches removed from the tracked
  active tree and preserved under ignored `.local_artifacts/`.

Nothing was scientifically erased: protocols, code, tests, reports, decisions, and lightweight
manifests remain in the archive.

## Retained active infrastructure

- `src/recoalign` OpenCLIP/retrieval benchmark and provenance system.
- Dataset/checkpoint manifests and schema validation.
- Winoground audit and cache-free comparison gates.
- Root tests and CI lifecycle.

## Added active modules

- `datasets/records.py`: canonical synthetic scene record.
- `synthetic_world/`: deterministic renderer, graph, relation vocabulary, full/partial/corrupted
  graph variants, and OOD composition generation.
- `models/vlm/`: `BaseVLM`, `ReferenceVLM`, and explicit LLaVA adapter boundary.
- `models/structure_encoder/`, `models/reasoning_interface/`, `models/alignment/`: future-proof
  contracts without speculative losses.
- `diagnosis/`: semantic probes, representation slices, and interface-gap contrasts.
- `evaluation/metrics.py`: condition accuracy, bootstrap CI, paired differences, JSON output.
- `experiments/`: Graph-vs-Text, graph ablation, OOD composition, and suite runner.
- YAML configs for synthetic, LLaVA, graph-vs-text, graph ablation, OOD, and benchmark runs.

## Scientific status

The active code validates the Structured Reasoning Interface Gap as an experimental question. It does
not claim a trained ReCoAlign method. The default backend is a deterministic reference backend for
pipeline validation; real LLaVA runs require an explicitly injected, pinned backend and checkpoint
manifest. The root `recoalign` CLI is the sole public experiment entry point.
