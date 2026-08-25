# Research Pivot: Minimal Falsification Experiments

## Purpose

This plan discriminates among four explanations of the frozen failures before any new model, loss,
adapter, or benchmark is introduced. The designs reuse the frozen synthetic world and retain all
EXP001–EXP004 evidence. `PIVOT_EXP_A`–`D` are planning identifiers, not registered experiments;
their protocols, power analysis, thresholds, and configs must be frozen before execution.

## Shared controls

- Use the same frozen scene generator, questions, answer evaluator, prompt version, and five-seed
  policy unless a preregistered power analysis requires more seeds.
- Keep the image present in all primary factorial cells.
- Match facts, entity references, answer options, and token budgets across serialization cells.
- Generate corruptions before inference, reject accidentally true corruptions, and retain their
  manifests.
- Use scene-disjoint train/test partitions for every probe.
- Report per-scene paired effects, seed-level uncertainty, and all failed integrity checks.
- Do not train ReCoAlign or modify the VLM during this stage.

## PIVOT_EXP_A — Semantic × Relation × Serialization factorial

### Hypothesis

PH001: compositional failure reflects joint semantic–relational integration limits conditioned by
representation format.

### Intervention

Construct a preregistered 2 × 2 × 2 factorial input while keeping the image and question fixed:

1. semantic evidence correctness: oracle object/attribute facts vs token-matched label-swapped facts;
2. relation evidence correctness: oracle relations vs token-matched relation-flipped facts;
3. serialization: fluent factual sentences vs canonical triples.

Include image-only and unchanged EXP001 caption/graph conditions as frozen reference anchors, not as
new decision cells. Both serializers must contain the same declared facts; wording templates and
corruption operators are fixed before inference.

### Primary estimands

- semantic correctness effect;
- relation correctness effect;
- semantic × relation difference-in-differences;
- serialization main effect and its interactions;
- effect by hop depth.

Use paired bootstrap confidence intervals over scenes within seed and a preregistered seed-level
replication rule. Numeric minimum effects must be chosen from power/sensitivity analysis without
examining new-model outcomes.

### Prediction

Semantic correctness and relation correctness are both positive, and their interaction is positive:
correct relations contribute most when entity/attribute semantics are also correct. Fluent
sentences remain at least as usable as triples; graph superiority is not predicted.

### Failure condition

Reject PH001 if any of the following holds:

- only semantic correctness matters and the controlled relation effect is absent;
- only serialization matters and semantic/relation correctness does not interact;
- the semantic × relation interaction confidence interval includes zero under the frozen gate;
- the pattern disappears under hop-depth or second-backbone replication;
- fact-equivalence, token-control, or corruption-validity assertions fail.

### Decision

This is the only candidate promoted to first execution. A GO authorizes independent replication and
construct-validity analysis, not model design or paper writing.

## PIVOT_EXP_B — Relation-selective grounding probe

### Hypothesis

PH002: relations are selectively missing from visual representations while object and attribute
semantics remain recoverable.

### Intervention

At identical frozen visual layers, train scene-disjoint probes for object identity, attribute,
single relation, ordered entity–relation binding, and random labels. Match label cardinality and
training examples, report chance-normalized accuracy, and compare linear with capacity-limited
nonlinear probes. Pair the probes with oracle-object-only and oracle-relation-only reasoning inputs.

### Prediction

Object and attribute decodability is high, relation and binding decodability is selectively low,
oracle object facts do not close the reasoning gap, and oracle relations do.

### Failure condition

Reject if all semantic categories are uniformly weak, if relation decoding is comparable to object
and attribute decoding, or if relation decoding is high while reasoning remains low.

## PIVOT_EXP_C — Primitive versus composition decodability

### Hypothesis

PH003: primitives are represented, but their bindings and novel compositions form a representation
bottleneck.

### Intervention

Using composition-disjoint splits, compare matched probes for primitive labels, ordered pairs,
relation bindings, two-relation chains, and answer programs. Control probe capacity, training-set
size, label entropy, and random features. Measure IID-to-OOD retention by hop depth.

### Prediction

Primitive probes succeed while binding/composition probes lose substantial OOD accuracy, with a
larger gap at greater hop depth.

### Failure condition

Reject if primitive and composition scores degrade together, if the gap vanishes after label/sample
matching, or if compositions are decodable but answer execution still fails.

## PIVOT_EXP_D — Cross-modal reasoning equivalence

### Hypothesis

PH004: task semantics can be represented visually, but the language reasoning path is poorly aligned
with image-derived evidence.

### Intervention

Present the same frozen semantic program as image-only evidence, fluent text, canonical triples, and
image-derived hidden context. Run only on samples/layers where the preregistered probe battery
establishes that required primitives are recoverable. Keep answer decoding identical and compare
modality gaps at matched semantic coverage.

### Prediction

Text reasoning remains strong while image-derived evidence underperforms despite adequate probe
availability; the gap is stable across relation types and depths.

### Failure condition

Reject if visual semantic probes are not adequate, text and visual conditions fail equally, or the
gap is explained by missing semantic coverage rather than cross-modal alignment.

## Candidate scoring

Scores range from 1 (weak) to 5 (strong) and are research-triage judgments, not scientific results.

| Candidate | Novelty | Evidence support | Feasibility | Q1 potential | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| PH001 Semantic–Structural Integration | 4 | 5 | 5 | 4 | **18** |
| PH002 Relational Grounding | 3 | 3 | 5 | 3 | 14 |
| PH003 Compositional Bottleneck | 4 | 2 | 3 | 4 | 13 |
| PH004 Cross-Modal Alignment | 3 | 4 | 4 | 3 | 14 |

PH001 is selected because it is the only candidate that directly accommodates all three eligible
observations without asserting an unmeasured internal mechanism: low registered SAS, caption over
graph, and correct over corrupted relations.

## Candidate causal chain

The following is a hypothesis to test, not a finding:

```text
limited visually grounded semantic availability
                    +
format-conditioned relational evidence consumption
                    ↓
semantic × relation integration failure
                    ↓
unreliable compositional reasoning
```

EXP004 motivates the first input, EXP001 motivates serialization conditioning, and EXP002 motivates
the relation-correctness input. Only PIVOT_EXP_A can test whether their interaction is causal.

## Execution order

1. Freeze PIVOT_EXP_A protocol, corruption validity, power analysis, and decision thresholds.
2. Run a non-scientific data-integrity preflight without model inference.
3. Execute LLaVA-1.5 across all registered seeds without adapting prompts after results.
4. If GO, replicate on one independently pinned eligible backbone.
5. If NO-GO, select the surviving candidate using the prespecified discrimination table; do not
   redefine PH001 post hoc.
6. Consider a method only after a replicated diagnostic hypothesis receives GO.
