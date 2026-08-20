# Benchmark evaluation protocol

## Categories and metrics

- Mechanism: EXP001 graph/text contrast, EXP002 structure necessity, EXP003 OOD
  composition retention and depth behavior.
- Compositional: SugarCrepe pairwise accuracy by object/attribute/relation/category;
  ARO multiple-choice accuracy; Winoground image/text/group accuracy; CREPE
  pairwise compositional accuracy.
- General reasoning: GQA and MMVP exact/normalized/multiple-choice accuracy.
- Capability preservation: paired general-QA accuracy and frozen Flickr30K/MS COCO
  retrieval metrics.

All generative predictions include sample ID, model, method, seed, ground truth,
category, compositional dimension, failure type, and oracle-input declaration.

## Statistical protocol

Pilot results require three shared seeds; paper claims require five. Mean, standard
deviation, percentile 95% CI, and paired bootstrap differences are generated from
the same sample/seed pairs. A single checkpoint or seed is never summarized as a
cross-model claim.

## Baselines

The core matrix contains Original VLM, Caption Reasoning, Oracle Graph Prompt upper
bound, and ReCoAlign. Existing methods enter only after official-code, license,
checkpoint-hash, and protocol-equivalence review. Oracle graph prompting is marked
as external upper-bound information and cannot be presented as ReCoAlign.

## Protocol lock

Dataset split, prompt, decoding, evaluator, and seed set are immutable across
methods. Test-set tuning, per-model prompts, and selective benchmark removal are
validation errors or visible blockers.
