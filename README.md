# ReCoAlign: Semantic–Structural Integration Diagnostics for Vision-Language Models

ReCoAlign is now in a research-pivot stage. Its first frozen LLaVA-1.5-7B evidence rejected the
original claim that sufficient visual semantics are preserved but a graph interface is missing.
The current candidate hypothesis is deliberately narrower:

> Compositional failure may reflect the joint limits of visually grounded semantic availability
> and serialization-conditioned integration of correct relational evidence. Correct relations can
> matter without graphs being a privileged reasoning interface.

This candidate is not yet a result. The repository is organized around falsifying it with minimal
diagnostic interventions before introducing any new loss, adapter, or trainable model.

The Phase-6 paper-package tooling is documented in
[`docs/paper_ready_package.md`](docs/paper_ready_package.md). It builds an auditable evidence map,
candidate frozen registry, reproducibility package, anonymous-submission files, and paper exports.
The first frozen LLaVA-1.5-7B claim-evidence execution is now complete for EXP001, EXP002, and
EXP004; EXP003 stopped at its preregistered token-length integrity gate after one full seed. The
scientific submission decision remains `NO-GO`: EXP001 and EXP004 falsified their registered gates,
EXP002 passed, and EXP003 is `INCONCLUSIVE`. Auditable metrics, decisions, figures, and compressed
raw predictions are published under [`reports/paper_evidence/`](reports/paper_evidence/README.md).
See [`reports/paper_readiness_report.md`](reports/paper_readiness_report.md) for the current decision.
The claim-level pivot audit is in [`research/evidence_audit.md`](research/evidence_audit.md), the
rejection record is in [`research/rejected_hypothesis.md`](research/rejected_hypothesis.md), and the
next falsification plan is in [`research/next_experiment_plan.md`](research/next_experiment_plan.md).

## Research boundary

- Frozen benchmark and retrieval infrastructure remains available only as a capability-preservation
  control; it is not the active scientific contribution.
- EXP001–EXP004 and all unfavorable evidence are retained as the immutable pivot basis.
- The Structured Reasoning Interface Gap is rejected as the active claim; it must not be rescued by
  post-hoc reinterpretation.
- The active line is preregistration of a semantic × relation × serialization falsification study.
- Paper writing and model design remain blocked.
- The default backend is a deterministic CPU reference backend for CI. LLaVA-1.5 is an optional
  injected backend selected by configuration and never silently substituted.

## Repository map

```text
configs/                         YAML experiment contracts
research/                        hypotheses, experiments, protocols, and decision policy
datasets/                        JSON-compatible scene records
models/
  vlm/                           BaseVLM, reference backend, LLaVA boundary
  structure_encoder/             retained graph intervention boundary
  reasoning_interface/           retained diagnostic dataclasses/protocols
  alignment/                     inactive legacy/future boundary (no current method claim)
diagnosis/
  semantic_probe/                availability probes
  representation_analysis/       slice summaries
  interface_gap_analysis/        structured-reasoning contrasts
src/recoalign/synthetic_world/
  ontology/                       finite semantic factors and typed relations
  generator/                      world-first deterministic generation
  renderer/                       reconstructable image rendering
  scene_graph/                    oracle graph and licensed inference
  questions/                      lossless captions and programmatic QA
  splits/                         IID and factor-held-out policies
  validation/                     schema, answer, leakage, and equivalence gates
experiments/
  graph_vs_text/                 Experiment A
  graph_ablation/                Experiment B
  ood_composition/               Experiment C
  benchmark/                     suite orchestration
evaluation/                      unified JSON metrics
runs/                            ignored, self-contained governed run bundles
training/                        future trainer protocol only
archive/                         registered NO-GO evidence and original Phase-3 code
src/recoalign/                   VLM, governance, synthetic-world, and preservation infrastructure
scripts/                         reproducible command wrappers
results/                         generated lightweight metrics and predictions
```

## Controlled synthetic world

Generate the validated 1,000-sample reference artifact and run the three-seed evaluation interface:

```bash
python -m recoalign generate-synthetic --config configs/synthetic_benchmark.yaml --count 1000
python -m recoalign evaluate-synthetic --config configs/synthetic_benchmark.yaml
```

The first command writes reconstructable image/scene/graph/metadata bundles. The second writes
`metrics.json`, `predictions.jsonl`, and `decision_report.yaml`; the shipped reference backend is
infrastructure-only and cannot produce a scientific GO decision. See
`docs/synthetic_world_design.md` and `docs/synthetic_dataset_protocol.md` for the controlled-factor,
information-equivalence, and leakage protocols.

## Governed historical mechanism experiments

The root `recoalign` CLI remains the canonical public entry point for reproducing the frozen
EXP001–EXP004 history. These commands do not constitute the new pivot experiment. The reference
backend requires only the base Python dependencies and is suitable for smoke tests:

```bash
python -m recoalign validate-research
python -m recoalign list-experiments
python -m recoalign run-experiment EXP001 --dry-run
python -m recoalign run-experiment EXP002 --dry-run
python -m recoalign run-experiment EXP003 --dry-run
```

Each governed invocation writes:

```text
runs/<experiment-id>/<run-id>/
├── config.resolved.yaml
├── command.txt
├── environment.txt
├── git_commit.txt
├── seed.txt
├── metrics.json
├── log.txt
├── manifest.json
├── decision_report.yaml        authoritative machine-readable decision
└── decision_report.md          human-readable mirror
```

To use LLaVA-1.5, install the optional Transformers/bitsandbytes environment, point
`configs/llava.yaml` at a locally verified checkpoint, and inject the matching backend into
`models.vlm.Llava15VLM`. The checkpoint revision and quantization must be recorded in the resolved
run manifest.

## Reproduce the baseline controls

The original OpenCLIP benchmark pipeline remains available through the root `recoalign` CLI. It
supports dataset manifests, checkpoint hashes, cache/no-cache verification, Winoground review, and
schema-validated reportability. These retrieval metrics are explicitly capability-preservation
controls for the structured-reasoning experiments, not the new scientific contribution. The frozen
identity and claim limits are documented in [`docs/research_identity.md`](docs/research_identity.md),
[`docs/contribution_framework.md`](docs/contribution_framework.md), and
[`docs/claim_boundary.md`](docs/claim_boundary.md).

```bash
python -m recoalign validate-config configs/baseline/openclip_vit_b32_laion2b_winoground.yaml
python -m recoalign run-baseline \
  --config configs/baseline/openclip_vit_b32_laion2b_winoground.yaml
```

Raw datasets, model weights, generated images, and feature arrays stay outside Git. See
[`docs/architecture.md`](docs/architecture.md), [`docs/reproducibility.md`](docs/reproducibility.md),
and [`archive/README.md`](archive/README.md) for the scientific and provenance boundaries.
Scientific governance details are in [`docs/experiment_governance.md`](docs/experiment_governance.md),
[`docs/hypothesis_tracking.md`](docs/hypothesis_tracking.md), and
[`docs/decision_protocol.md`](docs/decision_protocol.md).

## EXP001 controlled graph-vs-text validation

Run the complete five-seed information- and token-controlled instrument bundle with:

```bash
python -m recoalign run-graph-vs-text \
  --config configs/graph_vs_text.yaml \
  --output outputs/EXP001
```

The bundle includes paired predictions, natural/token-matched results, 1–4 hop metrics, error
taxonomy, graph/text/serialization ablations, preliminary composition-OOD results, three figures,
artifact hashes, and an eligibility-aware decision report. The shipped ReferenceVLM config validates
infrastructure only, so its decision is necessarily `INCONCLUSIVE`. See
[`docs/exp001_graph_vs_text.md`](docs/exp001_graph_vs_text.md) and the explicit local-checkpoint stage-2
template at [`configs/exp001_llava15.yaml`](configs/exp001_llava15.yaml).

## EXP002 structural-necessity validation

Run the complete five-seed graph completeness/corruption bundle with:

```bash
python -m recoalign run-graph-ablation \
  --config configs/graph_ablation.yaml \
  --output outputs/EXP002
```

EXP002 retains Image Only and Full Graph baselines; 25/50/75% supporting-relation removal; relation
flip, entity swap, and same-size Random Graph controls; format and opaque-label interventions; exact
Full/Random token matching; paired-bootstrap statistics; hop/relation tables; manifests; and three
figures. ReferenceVLM is again infrastructure-only. See
[`docs/exp002_structural_necessity.md`](docs/exp002_structural_necessity.md).

## EXP003 OOD compositional-generalization validation

```bash
python -m experiments.ood_composition.runner \
  --config configs/ood_composition.yaml \
  --output outputs/EXP003
```

EXP003 constructively generates IID, composition-novelty, unseen relation-combination, and 3–4 hop
OOD partitions. It compares Image, Caption, Graph, and length-matched Random Graph conditions and
reports generalization gap, retention, depth curves, paired bootstrap statistics, and three
anti-memorization slices. See [`docs/exp003_ood_composition.md`](docs/exp003_ood_composition.md).

## Phase 2.2 structured-interface diagnosis

```bash
recoalign run-interface-diagnosis --model reference
recoalign run-experiment --experiment EXP004 --dry-run
```

EXP004 measures Visual Semantic Availability (SAS), Structured Accessibility (StAS), and Reasoning
Execution (RES), plus oracle-graph gain, graph reconstruction/latent relation probes, consistency,
and the registered failure taxonomy. ReferenceVLM output is infrastructure validation only; missing
hidden-state or graph-reconstruction APIs are recorded as unavailable. See
[`docs/interface_diagnosis.md`](docs/interface_diagnosis.md),
[`docs/mechanistic_analysis.md`](docs/mechanistic_analysis.md), and
[`docs/failure_taxonomy.md`](docs/failure_taxonomy.md).

## Phase 2.1 real VLM evaluation

```bash
recoalign list-vlm-models
recoalign run-vlm-eval \
  --model llava_1_5_7b \
  --experiment EXP001 \
  --split test \
  --dry-run
```

The model registry exposes ReferenceVLM, LLaVA-1.5, LLaVA-NeXT, Qwen-VL, and InternVL through one
`BaseVLM` lifecycle. All backbones use the same versioned prompt and answer evaluator. LLaVA-1.5 is
the first pinned execution target; the other real-model rows remain adapter-ready until checkpoint
and code revisions are registered. See [`docs/vlm_integration.md`](docs/vlm_integration.md),
[`docs/vlm_evaluation_protocol.md`](docs/vlm_evaluation_protocol.md), and
[`docs/model_comparison_protocol.md`](docs/model_comparison_protocol.md).
