# EXP001: Graph vs Text Controlled Validation

## Motivation

EXP001 addresses the strongest information-confound objection to ReCoAlign: a graph condition may
appear better merely because it supplies more semantic facts. The experiment therefore tests
whether structure has reasoning value after declared semantic information and model-token budget
are controlled. It does not test whether graphs are better than pixels.

## Pre-implementation audit

The Synthetic Compositional World v2 already provided world-first generation, deterministic images,
four question families, 1–4 hop metadata, oracle graph answers, and a schema-validated equality hash
for caption and graph facts. The governance framework already provided linked H001/EXP001 records,
multi-seed aggregation, provenance, and an eligibility-aware decision engine.

The audit also found five blockers: the registered EXP001 dataset still pointed to generator-v1;
the runner was a generic four-condition placeholder; no tokenizer-level matched setting existed;
the requested error/depth/ablation/visualization outputs were absent; and the top-level required
`outputs/EXP001` artifact bundle was not generated. This implementation closes those engineering
gaps and explicitly amends EXP001 to the generator-v2 manifest before model evidence is collected.

## Hypothesis and falsification

H001 predicts that `Image + Scene Graph` outperforms `Image + Caption` when both carry the same
objects, attributes, and relations. The graph-interface route is falsified by an eligible complete
run if the overall advantage is absent, disappears after token matching, is limited to one hop, or
does not transfer to the preliminary composition-OOD slice.

## Implementation

The main implementation is split across:

- `src/recoalign/synthetic_world/questions/conditions.py`: controlled evidence serialization,
  semantic-unit accounting, token matching, graph order, unordered text, and graph serialization.
- `models/vlm/base.py`: model-neutral `prepare_input()`, `generate()`, and `evaluate()` lifecycle,
  including a tokenizer hook and auditable prepared-input metadata.
- `models/vlm/llava.py`: LLaVA-1.5 adapter using an injected pinned backend/tokenizer, with no silent
  fallback; `auto_load: true` explicitly enables a local-only lazy Transformers backend.
- `experiments/graph_vs_text/analysis.py`: task/depth accuracy, paired effects, token efficiency,
  information-control audit, and error taxonomy.
- `experiments/graph_vs_text/runner.py`: five-seed IID/OOD execution, governed aggregation, artifact
  manifests, plots, and GO/NO-GO/INCONCLUSIVE reporting.

ReferenceVLM remains a deterministic CPU test fixture. Its configured condition accuracies exercise
all branches but are not empirical observations. A real LLaVA run requires a pinned local checkpoint,
an injected runtime backend, and a preregistration amendment that marks that backend eligible before
execution.

## Conditions and controls

| Condition | Image | Objects/attributes | Relations | Purpose |
|---|:---:|:---:|:---:|---|
| C0 Image Only | yes | visual only | visual only | baseline |
| C1 Object List | yes | explicit | absent | object-information control |
| C2 Caption | yes | explicit | explicit | information-matched text |
| C3 Scene Graph | yes | explicit | explicit | structured intervention |

C2 and C3 equality is enforced by the same canonical-fact SHA-256. Each prediction records full
input tokens, evidence tokens, semantic units, relation count, serialization, padding count, and
residual token delta. Natural and token-matched results are always both retained.

## Evaluation and statistics

The four task families are object, relation, multi-hop, and attribute reasoning. Results include
accuracy at depths 1–4 and a three-class error taxonomy. Paired comparisons operate on identical
scene IDs. Seed-level effects are aggregated using mean, sample standard deviation, percentile
bootstrap 95% CI, one-sided paired t-test, and positive-seed fraction. Five seeds are required for
the critical run; three are allowed only for non-critical development tests.

The output bundle contains:

```text
outputs/EXP001/
├── config.resolved.yaml
├── metrics.json
├── predictions.jsonl
├── run.json
├── manifest.json
├── decision_report.yaml
├── decision_report.md
├── figure1_performance_comparison.png
├── figure2_reasoning_depth.png
├── figure3_token_efficiency.png
└── seeds/<seed>/...
```

## Decision criteria and interpretation

GO is allowed only when every registered overall, token, depth, OOD, significance, confidence, and
seed-robustness gate passes with an eligible frozen target VLM. Complete eligible evidence failing
any gate is NO-GO. Infrastructure fixtures, missing checkpoint evidence, registration mismatch, or
incomplete integrity checks yield INCONCLUSIVE.

Consequently, the shipped ReferenceVLM run answers only “is EXP001 executable and audited?” It cannot
answer “does structure help a VLM?” That scientific interpretation must wait for the preregistered
LLaVA-1.5 phase; failed results must remain in the same artifact bundle and must not be hidden.

The stage-2 execution template is `configs/exp001_llava15.yaml`. It loads only the declared local
checkpoint and fails loudly when the optional GPU stack or checkpoint is missing. This template is
intentionally not the currently registered scientific config: before target-model evidence is
collected, the EXP001 registry must be amended to the pinned LLaVA backend/config and independently
reviewed. Running an unamended template remains INCONCLUSIVE by construction.
