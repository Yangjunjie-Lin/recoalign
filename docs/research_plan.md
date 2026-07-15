# ReCoAlign research plan

## Problem statement

Global image-text contrastive objectives can reward coarse semantic overlap while underweighting distinctions involving attributes, relations, counts, object roles, and word order. ReCoAlign will test whether typed hard negatives and local visual evidence improve these distinctions while preserving conventional retrieval performance.

## Phase 1 — Baseline integrity

1. Freeze the dataset splits and evaluation protocol.
2. Reproduce zero-shot OpenCLIP ViT-B/32 retrieval.
3. Validate bidirectional R@1, R@5, R@10, and mean recall.
4. Add compositional benchmark adapters without training on benchmark test sets.
5. Record runtime, peak memory, dependency versions, and random seeds.

Exit criterion: independently repeatable baseline tables with no method contribution.

## Phase 2 — Compositional negatives

1. Define a perturbation taxonomy: object, attribute, relation, count, role, and order.
2. Preserve the source caption and provenance for every generated negative.
3. Filter malformed, semantically equivalent, and visually unsupported negatives.
4. Compare training-time contrastive use against inference-time reranking.
5. Audit performance separately for each perturbation type.

Exit criterion: gains cannot be explained only by templates or benchmark artifacts.

## Phase 3 — Region–phrase alignment

1. Establish a patch-token baseline before adding external grounding models.
2. Align noun phrases and relation phrases with local visual evidence.
3. Compare max, attention-weighted, and optimal-transport-style aggregation.
4. Test whether local alignment complements rather than replaces global similarity.
5. Produce qualitative maps and failure cases using fixed selection rules.

Exit criterion: statistically and qualitatively supported improvement over the hard-negative baseline.

## Phase 4 — Journal-grade validation

- multiple backbones and pretrained checkpoints;
- multiple random seeds for trained variants;
- standard retrieval and compositional benchmarks;
- complete component and loss-weight ablations;
- efficiency, memory, and parameter analysis;
- robustness and transfer experiments;
- honest negative results and limitations.

The target venue should be selected after the contribution and evidence are known, not used to overstate preliminary work.
