# Research Pivot Evidence Audit

## Audit decision

The frozen LLaVA-1.5-7B evidence rejects the original Structured Reasoning Interface Gap as the
active explanation. The evidence supports a narrower observation—that correct relational evidence
matters once supplied—but does not show that graphs are a privileged interface or that visual
semantics are already preserved at the required level.

## Claim-level audit

### Claim: H001 — A graph interface improves reasoning over information-controlled text

**Evidence:** EXP001 completed five registered seeds and 8,000 predictions. Natural Graph − Caption
was −0.0433, 95% CI [−0.0583, −0.0283]. Token-matched Graph − Caption was −0.0783, 95% CI
[−0.1017, −0.0550]. Every registered criterion failed in 5/5 seeds.

**Status:** FALSIFIED.

**Interpretation:** Graph serialization is not a naturally superior reasoning interface for the
frozen LLaVA model. Because the benchmark manifest records caption/graph fact equivalence, the
current evidence supports a representation-format or model-compatibility difference; it does not
yet support the post-hoc explanation that the graph simply omitted more facts.

### Claim: H002 — Performance depends on correct and complete relational evidence

**Evidence:** EXP002 completed five seeds and 6,000 predictions. Full − Random Graph was +0.4517,
95% CI [+0.4400, +0.4617], and Full − Wrong Graph was +0.1800, 95% CI [+0.1583, +0.2033]. All seven
registered criteria passed in every seed.

**Status:** SUPPORTED within the frozen LLaVA-1.5 and synthetic-world scope.

**Interpretation:** The model uses the content and correctness of externally supplied relational
evidence. This result establishes relation sensitivity, not graph optimality, autonomous graph
construction, an internal interface gap, or ReCoAlign efficacy.

### Claim: H003 — A graph advantage persists under compositional OOD

**Evidence:** EXP003 produced 1,920 first-seed predictions, but the preregistered
`random_graph_length_matched` assertion failed before multi-seed promotion.

**Status:** INCONCLUSIVE; the old graph-advantage hypothesis is RETIRED rather than declared
falsified.

**Interpretation:** The retained first-seed values are diagnostic only. The protocol cannot support
an OOD claim, and the failure does not authorize changing the integrity gate after seeing results.
Because EXP001 removed the prerequisite graph advantage, repairing EXP003 is not the next research
priority.

### Claim: H004 — Semantic availability is high but structured accessibility is low

**Evidence:** EXP004 completed five seeds and 360 predictions. SAS was 0.1214, StAS was 0.2931,
RES was 0.8500, and SAS − StAS was −0.1717, 95% CI [−0.2073, −0.1360]. The registered classifier
selected `TYPE_A_SEMANTIC_FAILURE`.

**Status:** FALSIFIED for the registered LLaVA diagnosis.

**Interpretation:** The required high-SAS/low-StAS pattern is absent. Low SAS means the registered
probe did not recover sufficient task semantics; it does not by itself prove that all semantic
information is absent from every layer or inaccessible to every nonlinear decoder.

### Claim: Correct relational information matters for compositional reasoning

**Evidence:** Correct graphs outperform partial, random, wrong, and relation-label-randomized
graphs in EXP002, with stable positive effects across five seeds and increasing depth dependency.

**Status:** SUPPORTED, narrowly scoped.

**Interpretation:** Relation correctness is causally relevant within the supplied-evidence
intervention. The claim must not be expanded to “graph is the missing interface.”

### Claim: Caption superiority reveals why graphs fail

**Evidence:** EXP001 establishes Caption > Graph under natural and token-matched conditions.

**Status:** OBSERVATION VERIFIED; mechanism UNRESOLVED.

**Interpretation:** Fluent-language compatibility, serialization familiarity, graph parsing cost,
and probe/task artifacts remain competing explanations. The existing experiment does not identify
which explanation is causal.

### Claim: ReCoAlign learns or repairs the failure

**Evidence:** No claim-eligible trained real-VLM ReCoAlign checkpoint or paired method comparison is
frozen.

**Status:** UNSUPPORTED / INFRASTRUCTURE ONLY.

**Interpretation:** No method or architecture claim is permitted during the pivot stage.

## What survives the pivot

- the frozen synthetic world, equivalence controls, and seed protocol;
- EXP002 as evidence that relation correctness matters;
- EXP001 and EXP004 as retained falsifying evidence;
- EXP003 as an unpromoted integrity failure;
- SAS/StAS/RES as diagnostic instruments whose construct validity still requires controls;
- all checkpoint, prediction, metric, and provenance records.

## What no longer survives

- “VLMs preserve semantics but merely lack a graph interface” as an active claim;
- “graph is the missing intermediate representation”;
- graph superiority or OOD benefit;
- any learned ReCoAlign method claim;
- paper-writing readiness.
