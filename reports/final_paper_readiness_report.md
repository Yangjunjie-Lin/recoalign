# ReCoAlign Final Paper Readiness Audit

## Decision

**NO-GO**

Do not begin claim-bearing paper writing. The repository is engineering-ready and its research
identity is frozen, but the evidence required for the core paper claims is not complete.

## 1. Current status

Audit target:

- branch: `agent/publish-paper-results`
- audited commit: `c39d5f411ef7139c90a7d68205a36c63a737aaaa`
- target branch: `main`
- CI at audit start: Python 3.10 and 3.11 passed

### Executive evidence summary

| Gate | Observed state | Audit result |
|---|---|---|
| Research identity | Structured Reasoning Interface Learning is frozen and consistently bounded | PASS |
| Claim evidence | `0/7` claims verified; four infrastructure-only and three pending | FAIL |
| Comprehensive evaluation | `0/432` registered cells complete | FAIL |
| Real VLM mechanism evidence | LLaVA smoke/incomplete attempt only; other families are adapter boundaries | FAIL |
| Learned method evidence | Toy optimization and checkpoint round-trip only; no claim-eligible trained checkpoint | FAIL |
| Mechanistic ablations | 14 interventions registered, real multi-seed evidence pending | FAIL |
| Tracked run artifacts | 0 prediction files, 0 metrics files, 0 run metadata files, 0 checkpoint weight files | FAIL |
| Reproducibility framework | Environment/protocol/manifests present; claim-bearing per-run bundles absent | PARTIAL |
| Paper package | Reports/figures/tables scaffolding exists, but the requested complete evidence package does not | PARTIAL |

The machine-readable repository status agrees with this audit:

- `docs/evidence_map.yaml`: `verified: 0`, `infrastructure_only: 4`, `pending: 3`;
- `reports/submission_readiness.yaml`: `scientific_submission_decision: NO-GO`;
- `reports/integrity_report.yaml`: `scientific_submission_ready: false`;
- `reports/comprehensive_evaluation_report.md`: 0 observed and 0 complete cells;
- `reports/mechanistic/decision_report.yaml`: real-VLM mechanistic evidence pending.

### Research identity audit

The active README, research plan, architecture, frozen hypothesis, contribution framework, and
claim boundary consistently describe:

```text
visual semantic availability
        ↓
structured representation accessibility
        ↓
compositional reasoning execution
```

Retrieval/OpenCLIP modules and configs remain in the repository, but the active documents label
them as capability-preservation controls or historical compatibility infrastructure. They are not
presented as the ReCoAlign contribution. No graph-prompt result is described as proof that the
learned interface works. The identity migration therefore passes.

### Scientific claim audit

| Paper claim | Required evidence | Observed evidence | Result |
|---|---|---|---|
| Structured Reasoning Interface Gap exists | Claim-eligible EXP001–EXP003 results | Protocols and ReferenceVLM infrastructure only | FAIL |
| Visual semantics are preserved | Real-model representation diagnosis | EXP004 real hidden-state access is pending; current diagnosis is infrastructure-only | FAIL |
| ReCoAlign learns the missing interface | Trained interface checkpoint and real-VLM method runs | Toy training/round-trip only; no tracked trained checkpoint | FAIL |
| Improvement generalizes | Complete multi-model benchmark evidence | 0/432 cells complete | FAIL |

Because every core claim lacks claim-eligible evidence, the decision cannot be upgraded on the
strength of implementation completeness.

### Mechanism experiment audit

- **EXP001 — Graph vs Text:** the config and protocol include information control, token matching,
  paired statistics, multi-hop slices, and fixed decision criteria. No complete eligible real-VLM
  multi-seed result is present.
- **EXP002 — Structural Necessity:** full, partial, corrupted, randomized, token-length, and label
  controls are registered. Existing evidence is ReferenceVLM infrastructure validation, not a
  scientific result.
- **EXP003 — OOD Composition:** composition-disjoint splits, primitive coverage, and leakage
  assertions are registered and tested. No claim-eligible real-VLM OOD result is present.

The experiment designs are substantially complete. The experiment evidence is not.

### Method completeness audit

Implemented components include structure-token banks, visual-to-structure encoders, alignment and
adapter modules, three losses, staged trainers, optimizer/scheduler/checkpoint utilities, toy
training, resume tests, and registered ablations.

The claim-bearing method path remains incomplete:

- `checkpoints/` contains documentation only; no trained ReCoAlign weights are tracked;
- LLaVA exposes a model-level injection boundary, but its registered backend does not provide a
  completed learned-token hidden-context runtime;
- LLaVA-NeXT, Qwen-VL, and InternVL remain adapter boundaries without pinned executable method
  runtimes;
- no real VLM has a complete ReCoAlign training → checkpoint → inference → evaluation chain.

This is a real implementation scaffold and toy model, not merely empty files, but it is not yet a
complete paper method execution.

### Real VLM validation audit

LLaVA-1.5 loaded and produced a single-sample smoke result locally. A registered EXP001 attempt
reached 629/1,600 cached requests before interruption; the repository correctly retained it as
incomplete and did not promote partial metrics. There is no complete LLaVA EXP001–EXP003 result.

LLaVA-NeXT, Qwen-VL, and InternVL have unified adapter/config boundaries but no claim-eligible
execution. The requirement for a primary model plus at least one additional VLM is not met.

### Ablation audit

The repository registers the required ablation families:

- no structure interface;
- random structure tokens;
- fixed graph encoder/oracle graph comparison;
- semantic, structural, and reasoning loss removals;
- parameter-matched capacity control;
- remove, shuffle, and cross-sample token interventions.

However, the real multi-seed ablation matrix is unexecuted. The mechanistic report explicitly marks
every claim-bearing criterion as pending. Registration is not evidence.

### Result integrity audit

At the audited commit, the Git tree contains no complete run-level `predictions.jsonl`,
`metrics.json`, `run.json`, or model weight file. Local ignored synthetic outputs may exist in a
developer workspace, but they are not branch artifacts and cannot support this audit.

The generated paper tables contain rows marked `planned` or `blocked`, generally with zero seeds
and blank accuracy. The statistics files contain no complete cell summaries or paired comparisons.
Thus no paper table currently has the required config + checkpoint + seeds + metrics + raw
predictions chain.

### Reproducibility and paper-package audit

The repository has strong reproducibility infrastructure: environment specifications, dataset and
checkpoint manifests, frozen configs, prompt versions, run schemas, and governance gates. These
assets are sufficient to execute and audit future experiments.

They do not replace missing run evidence. The exact requested `paper_package/` hierarchy is absent.
Equivalent scaffolding exists under `reports/`, `reproducibility/`, and `submission/`, including ten
figures and three report tables, but experiment summaries and appendix result bundles backed by
complete runs are absent. Current exports deliberately render missing evidence as pending.

## 2. Missing components

1. Complete LLaVA-1.5 EXP001, EXP002, and EXP003 under the frozen protocols and registered seeds.
2. Complete the same mechanism validation on at least one additional executable VLM family.
3. Run real-model EXP004 semantic, structured-accessibility, and reasoning-execution diagnosis.
4. Implement and verify learned-token hidden-context injection in the real backbone runtime.
5. Produce at least one claim-eligible trained ReCoAlign checkpoint with its config, hash, training
   log, environment, and dataset manifest.
6. Execute full, no-structure, random-token, parameter-matched, loss, supervision, oracle, and
   causal-intervention ablations over the frozen seed set.
7. Populate the multi-model benchmark matrix, including capability-preservation results and paired
   statistics; keep failed cells visible.
8. Publish each paper result with raw predictions, metrics, run metadata, config, seed, dataset
   identity, and checkpoint identity.
9. Rebuild the paper package only from complete, reportable runs and repeat this audit.

## 3. Reviewer attack points

- **Synthetic-only mechanism:** the current evidence cannot establish that real VLMs exhibit the
  claimed interface gap.
- **Graph augmentation alternative:** without learned-token real inference, reviewers can argue
  that observed graph effects only show oracle information augmentation.
- **Semantic-failure alternative:** real representation probes have not shown that the relevant
  visual semantics are preserved.
- **No method-efficacy evidence:** toy loss reduction and checkpoint round-trip do not demonstrate
  a useful ReCoAlign interface in a VLM.
- **No cross-model generality:** additional backbones are interfaces rather than executed evidence.
- **No causal attribution:** ablations and interventions are registered but not run on the target
  method/backbones.
- **No paper-result provenance:** tables are plans, not values traceable to raw predictions and
  checkpoints.
- **Premature paper framing:** writing a claim-bearing manuscript now would force the narrative to
  outrun the retained evidence.

## 4. Next action

Do not start the Introduction/Method/Experiments manuscript as if the scientific claims are
established. Begin the evidence-execution phase:

1. secure a CUDA environment capable of completing the frozen LLaVA runs;
2. finish and audit EXP001–EXP003 before adapting prompts, splits, or thresholds;
3. execute EXP004 and use its outcome to decide whether method training remains scientifically
   justified;
4. complete real hidden-context injection and TRAIN001–TRAIN003 for a claim-eligible checkpoint;
5. execute the registered ablation and benchmark matrices;
6. regenerate the evidence map and require nonzero verified claims before another readiness audit.

Merging this branch into `main` is acceptable as an engineering/research-governance milestone. It
must not be interpreted as approval to begin claim-bearing paper writing or as a scientific GO.
