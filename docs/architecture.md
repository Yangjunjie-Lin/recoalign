# Structured Reasoning Interface architecture

ReCoAlign separates stable benchmark infrastructure from the mechanism-validation path.

```text
image / caption
      │
      ▼
BaseVLM.encode_image / encode_text
      │
      ▼
VisualRepresentation
      │
      ▼
StructureEncoder → StructuredRepresentation (nodes + typed edges)
      │
      ▼
ReasoningRequest → LLMReasoning / BaseVLM.reason
      │
      ▼
answer + confidence + provenance
```

## Design rules

1. The visual encoder, structure encoder, and language reasoner are separate interfaces.
2. Synthetic conditions are paired at the scene level so gains are evaluated with matched contrasts.
3. Graph prompts must declare whether they are full, partial, or corrupted; an oracle graph is an
   intervention, not evidence that a model autonomously builds the graph.
4. No trainable loss is added until the interface gap is reproduced across seeds, backbones, and
   controlled graph ablations.
5. Every experiment uses YAML configuration, explicit seeds, JSON metrics, prediction rows, and a
   run manifest.
6. The root `recoalign` CLI is the only public experiment entry point; runner modules are library
   implementation details.

## Stable infrastructure

`src/recoalign/` is the authoritative package for benchmark records, retrieval metrics,
dataset/checkpoint manifests, environment capture, Winoground reportability, and the controlled
synthetic-world instrument. Root-level Phase-1 packages remain compatibility boundaries for existing
registered runners.

## Active Phase-1 path

- `src/recoalign/synthetic_world/` generates world state before deriving images, graphs, captions,
  and programmatic compositional questions.
- `models/structure_encoder/` defines the visual-to-graph boundary.
- `experiments/graph_vs_text/` compares image, object list, caption, and graph conditions.
- `experiments/graph_ablation/` measures graph completeness and corruption.
- `experiments/ood_composition/` evaluates compositions held out from the generator's train split.
- `diagnosis/interface_gap_analysis/` reports sufficiency contrasts without overclaiming a unique
  internal bottleneck.

## Future model boundary

`models/reasoning_interface/` reserves the contracts needed for Structure Token Learning,
ontology-guided representation, and graph-aware alignment. It intentionally contains interfaces and
data contracts only; the repository does not present a speculative model as a result.
