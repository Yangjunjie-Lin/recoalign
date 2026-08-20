# Next experiment roadmap

## Experiment A — Graph vs Text

Run `configs/graph_vs_text.yaml` with at least three seeds and the real frozen VLM backend. Compare:

- image only;
- unordered object list;
- generated caption;
- explicit scene graph.

Report paired accuracy, confidence intervals, relation slices, and exact prompt/provenance hashes.

## Experiment B — Graph completeness

Run `configs/graph_ablation.yaml` with:

- full graph;
- partial graph with one local edge removed;
- corrupted graph with one typed relation replaced.

Add graph-size and edge-order controls before interpreting a gain as structure-specific.

## Experiment C — OOD composition

Run `configs/ood_composition.yaml` with disjoint train/test composition signatures. Report in-domain
and unseen-composition performance for image, caption, and graph conditions, with per-seed paired
statistics.

## Required before any model contribution

1. Replicate all three experiments across at least two frozen VLM backbones.
2. Separate graph sufficiency from graph-construction ability using predicted, noisy, and oracle graphs.
3. Add caption-length, token-order, and relation-frequency controls.
4. Keep retrieval benchmarks as independent preservation controls.
5. Only then define a minimal structure-token or ontology-guided alignment intervention.

The current roadmap intentionally postpones new loss design until the interface gap survives these
controls.
