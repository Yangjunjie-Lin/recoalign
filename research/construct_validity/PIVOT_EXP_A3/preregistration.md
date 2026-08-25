# PIVOT_EXP_A3 Preregistration

## Governance and immutable parent

PIVOT_EXP_A2 remains frozen at commit `765d502dfdbdabd143945a08591b067cddbf8b3e` with outcome
`INCONCLUSIVE`. Its 5,300 predictions include 137 frozen manipulation-check `no_match` outputs.
PIVOT_EXP_A3 will not delete, reinterpret, rescore, or rerun any A2 row. The legacy parse-failure
audit is descriptive measurement-design evidence only. The separate continuation-scope record
authorizes a new construct-validity study, not a change to the A2 decision.

No real VLM inference may begin until this preregistration, the power analysis, source inventory,
trial inventory, tokenizer contract, parser tests, legends, hashes, and all preflight assertions
are frozen and passing. The frozen backend's no-weight runtime dry-run must also report all required
dependencies, compatible loader, processor, checkpoint, hardware, and CUDA readiness. Tokenizer
loading and synthetic-image construction are preflight, not VLM inference. Development accuracy
may never be computed or used to change M1/M2.

## Research question and constructs

The study distinguishes three constructs rather than treating one manipulation-check average as a
valid instrument:

1. **CV1 answer-contract comprehension** tests only the mapping between displayed alternatives and
   numeric option IDs. It uses the one frozen neutral image. For either oracle or corrupted
   scaffolds, the registered answer follows the currently displayed scaffold, never latent scene
   truth.
2. **CV2 semantic-scaffold comprehension** tests entity-to-shape, entity-to-color,
   shape-plus-color-to-entity, and entity-to-complete-description lookup under the neutral image.
   Ground truth follows the active scaffold. Each task is reported and gated separately.
3. **CV3 semantic rescue against scene truth** uses the original scene and scores the same four
   tasks against immutable scene truth. Oracle and corrupted scaffolds are paired within scene.
   Only CV3 can establish that a manipulation is sufficient for a future causal experiment.

This study cannot support a semantic-primary, integration-independent, or joint mechanism claim.

## Frozen model and answer contracts

The only backbone is `liuhaotian/llava-v1.5-7b` revision
`4481d270cc22fd5c4d1bb5df129622006ccd9234`. Training, gradient updates, adapters, losses, and model
parameter changes are prohibited.

The primary response is conditional log likelihood over exactly four option IDs: `1`, `2`, `3`,
and `4`. The prompt ends with the exact ASCII prefix `FINAL_CHOICE=` and no trailing space. The
registered tokenizer encodes the four continuations as distinct single tokens. The implementation
computes all four next-token log probabilities and selects their argmax without receiving ground
truth. A finite four-score vector always yields one registered prediction; ties resolve by numeric
registered order and are retained. There is no unparsed primary state. The primary method is fixed
as conditional likelihood; constrained decoding is not a result-contingent fallback.

The secondary response is deterministic free generation with grammar:

```regex
^\s*FINAL_CHOICE\s*=\s*([1-4])\s*$
```

The parser is ASCII-delimiter, uppercase, case-sensitive, and does not Unicode-normalize. It
rejects sentences, traces, aliases, duplicates, invalid IDs, substrings, and truncated fields.
No synonym or parser rule may be added after inference. Free generation is external-validity
evidence only and cannot replace forced-choice results.

## Semantic manipulations

Exactly two selectable candidates and one historical anchor are registered.

**M0 — Historical Anchor.** The A2-style canonical JSON representation with `Entity A` through
`Entity D` is retained only as a descriptive anchor. M0 can never be selected for PIVOT_EXP_A4.

**M1 — Canonical Entity Table.** The representation is fixed as:

```text
ENTITY_TABLE
E1 | shape=square | color=red
E2 | shape=circle | color=blue
E3 | shape=triangle | color=green
E4 | shape=sphere | color=yellow
```

Entity IDs are assigned to source objects by a scene-bound hash. Row order is separately randomized
by a scene-bound hash and never uses spatial order. The table has no coordinates, relations, query
roles, reasoning, or answer. Numeric answer IDs are a distinct namespace from `E1`–`E4`.

**M2 — Visual Object-Legend Scaffold.** Each legend row contains a stable entity ID, one fixed-size
standardized object rendering, shape, and color. Because the synthetic generator exposes no frozen
object bounding boxes, the preregistered crop replacement is a deterministic isolated re-render of
the source object with the same frozen synthetic renderer on a 96×96 neutral tile. Switching to
bounding-box crops or another fallback after preflight is prohibited. This replacement removes
absolute scene position while preserving standardized visual appearance. Row order is hash-random,
not left-to-right or top-to-bottom scene order. Legend canvases have fixed 576×384 pixels.

Oracle and corrupted scaffolds retain the same entity IDs, row order, entity count, dimensions,
and complete attribute multiset. Corruption applies one complete shape-color-object tile
derangement across entity IDs. Every corrupted binding must be false; partial or identity
permutations are rejected. No manipulation contains a relation, answer, query role, or reasoning
chain.

## Sampling, development, and validation freeze

The frozen synthetic-world generator `recoalign.synthetic_world.v2` is reused with new seeds.
Development seed `20260830` is separate from validation seeds `20260901`–`20260905`, and all are
separate from A2 seeds `20260818`–`20260822`. Candidate scenes are generated without model access.
Within each validation seed, exactly the first 100 scene-ID-sorted records containing exactly four
objects are selected from 1,200 generated candidates. The rule is fixed before generation outcome
inspection and provides four genuine within-scene alternatives for every task.

The development inventory has eight scenes and may be used only for schema, image, crop, tokenizer,
parser, prompt, and balance checks. No development response accuracy is permitted. Validation
scene IDs, seeds, questions, answers, alternative mappings, images, legend images, trial keys, and
SHA-256 hashes are frozen before the model is loaded for scoring.

## Trials and alternatives

Every semantic trial is four-way choice. Shape and color alternatives are the four genuine values
present in the scene. Object-identity alternatives are the four active entity IDs. Complete-binding
alternatives are the four genuine scene shape-color descriptions. There are no filler strings.

The target is the second registered entity (`Entity B` for M0; `E2` for M1/M2), whose source object
is scene-hash randomized. Corruption moves its complete binding. For object identity, the question
uses that target's true shape-color pair; the oracle declared answer is the target ID, while the
corrupted declared answer is the other entity to which the pair moved. Thus neutral corrupted CV2
is scored against what the current scaffold declares, while original-image corrupted CV3 is scored
against scene truth.

Alternative position is assigned without model outputs. Within every seed × task × manipulation,
each scene-truth position and each active-declaration position occurs exactly 25 times among the
100 scenes. Oracle and corrupted mappings are jointly scheduled so both targets are balanced;
each mapping is then reused across image and response cells. Distractor order within the remaining
positions is scene-hash randomized.

The primary forced-choice factorial crosses M0/M1/M2 × oracle/corrupted × neutral/original × four
tasks on 500 scenes (24,000 trials), plus 3,000 CV1 contract trials. The secondary matched subset is
limited prospectively to M1/M2 × oracle × neutral/original × four tasks (8,000 trials), plus 1,000
CV1 trials. Total planned predictions are 36,000: 27,000 primary and 9,000 secondary. M0 is not in
the secondary subset because it is neither selectable nor required to validate the new grammar.

## Registered gates

All gates apply to M1 and M2 separately. M0 is descriptive.

- **Gate A — Primary Answer Validity:** forced-choice valid measurement rate equals 1.00 exactly.
  CV1 option-mapping comprehension must also reach mean accuracy 0.95 with 95% CI lower bound
  0.90 separately under oracle and corrupted active scaffolds; corrupted answers follow the
  current displayed declaration.
- **Gate B — Secondary Parser Integrity:** parse rate is at least 0.99 and its 95% CI lower bound is
  at least 0.98.
- **Gate C — Scaffold Comprehension:** under neutral image plus oracle scaffold, each of shape,
  color, object identity, and entity binding has mean accuracy at least 0.95 and 95% CI lower bound
  at least 0.90.
- **Gate D — Scene Semantic Sufficiency:** under original image plus oracle scaffold, each task has
  mean accuracy at least 0.90 and 95% CI lower bound at least 0.85.
- **Gate E — Intervention Separation:** for original-image scene-truth scoring, each task has paired
  oracle-minus-corrupted accuracy at least 0.50 with 95% CI lower bound strictly above zero; the
  corrupted condition must not independently pass Gate D.
- **Gate F — Image Interference:** for oracle scaffolds, neutral-minus-original accuracy must pass
  paired TOST equivalence within ±0.05. A non-significant difference is not equivalence.

Every endpoint is task-specific. No overall average can override a task failure. Pooled confidence
intervals use a hierarchical percentile bootstrap that resamples five seeds and then paired scenes,
with 10,000 replicates and seed `20260826`. Gate-E superiority tests and Gate-F maximum TOST
p-values form separate four-task Holm families. Seed replication requires all five seeds present
and the registered point criterion in at least four seeds for every task. No seed or failed trial is
deleted.

## Power

The sample size is set by the most demanding endpoint, Gate-F equivalence. With 500 paired scenes,
true difference zero, paired discordance 0.10, margin ±0.05, and the worst first Holm threshold
0.0125, fixed-seed Monte Carlo power is 0.8028. Gate C, D, and E design alternatives exceed 0.999
planned power. The 9,000 secondary generations give effectively complete planned power for a true
parse rate 0.995 against the registered 0.99/0.98 gate. These design alternatives are not estimated
from new model answers.

## Selection and decision

M1 is selected if it passes A–F. M2 is selected only if M1 fails and M2 passes A–F. If both pass,
M1 is selected by the minimum-intervention principle. If both fail, M1 and M2 cannot be combined,
no M3 may be added, and prompts or thresholds cannot be changed after validation.

The only outcomes are `CONSTRUCT_VALIDITY_GO_TEXT`, `CONSTRUCT_VALIDITY_GO_VISUAL_LEGEND`,
`ANSWER_CONTRACT_FAILURE`, `SEMANTIC_MANIPULATION_FAILURE`, `IMAGE_INTERFERENCE_DIAGNOSIS`,
`NO-GO_CONSTRUCT`, and `INCONCLUSIVE`. A GO authorizes only PIVOT_EXP_A4 preregistration. Independent
backbone replication, model development, and paper writing remain prohibited under every outcome.
