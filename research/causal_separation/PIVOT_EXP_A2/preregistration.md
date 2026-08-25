# PIVOT_EXP_A2 Preregistration

## Research question

After an externally randomized scaffold makes object identity, shape, color, and entity binding
available, do relation correctness and representation format retain controlled effects on frozen
LLaVA-1.5 compositional decisions?

This study tests conditional causal effects of input interventions. It does not train a model,
select scenes by model behavior, expand the benchmark, or authorize a paper claim.

## Parent evidence and immutable boundary

PIVOT_EXP_A is fixed at tag `pivot-exp-a-v1`. Its 2,000 predictions, metrics, protocols, decision,
and artifact manifest are immutable inputs. PIVOT_EXP_A2 uses only existing EXP001 synthetic scenes
from the same five seeds.

## Sampling and exclusion

The primary matrix contains all 90 non-`relation_reasoning` scenes per seed: 30 object, 30
attribute, and 30 multi-hop questions. The multi-hop inventory contains ten scenes at each of
depths 2, 3, and 4. One-hop object and attribute questions provide 60 primary scenes per seed.

Single-hop relation-answer questions are excluded from the primary matrix before inference because
injecting their only supporting relation would directly state the answer. They may enter the
behavior-independent manipulation-check sample and native descriptive anchors, but never E1–E6.
No scene is selected using A1 correctness or any other model outcome.

## Randomized intervention

Each primary scene appears in all eight cells of a 2 × 2 × 2 within-scene design:

- semantic correctness: oracle or entity-attribute-deranged;
- relation correctness: registered supporting edges or a false, count-preserving corruption;
- serialization: canonical JSON or canonical triples.

Aliases are assigned by a seed- and scene-bound hash before inference. The oracle scaffold contains
only neutral entity references, shape, and color. It contains no query role, target relation,
reasoning chain, or explicit answer field. The corrupted scaffold deranges complete shape-color
bindings across the same entities and is rejected unless every new binding is false.

Correct relation evidence contains only registered supporting edges. Corruption preserves aliases
and edge count and is rejected if any corrupted edge is a true scene edge. JSON and triples encode
the exact same canonical fact inventory. All eight complete prompts are matched within one actual
LLaVA token using punctuation-only padding. Natural language and image-only cells are frozen
secondary references and cannot replace the primary JSON–triples comparison.

## Manipulation check

Twenty scenes per seed are selected by question-type-stratified scene ID, independently of behavior.
Four non-relational questions test object identity, shape, color, and entity-attribute binding under
oracle and corrupted scaffolds. The manipulation passes only when oracle mean accuracy is at least
0.90, its 95% CI lower bound is at least 0.85, and the paired oracle-minus-corrupted gain is at least
the registered superiority SESOI with CI lower bound above zero. Failure makes the entire study
`INCONCLUSIVE`; relation and format effects will not be interpreted.

## Estimands

- E1 semantic rescue: oracle minus corrupted semantics at fixed relation and format cells.
- E2 relation under semantic sufficiency: correct minus corrupted relation under oracle semantics.
- E3 format under semantic sufficiency: JSON minus triples under oracle semantics and correct relation.
- E4 semantic × relation difference-in-differences, averaged across primary formats.
- E5 semantic × format difference-in-differences under correct relation.
- E6 semantic × relation × format interaction as a secondary mechanism estimand.
- E7 all registered contrasts by hop depth 1, 2, 3, and 4.

E1 is a gated rescue test. E2 and E3 form the Holm-corrected primary conditional-effect family.
Scene-paired hierarchical bootstrap resamples seeds and scenes. Seed replication requires the
registered direction in at least four of five seeds. Mechanism conclusions additionally require
the registered direction in at least four of five seeds at both 2-hop and 3-hop.

## SESOI, power, and equivalence

The superiority SESOI and TOST equivalence margin are both 0.10 absolute accuracy. This deliberately
stricter-than-default threshold was chosen before inference because the study asks which residual
effects are large enough to motivate a costly independent-backbone replication. Frozen PIVOT_EXP_A
effects were 0.185–0.415; an effect below ten percentage points is not treated as mechanism-defining
in this single-backbone study. The 450-scene paired design has at least 0.80 planned power for the
gated rescue effect and Holm-controlled E2/E3 family under frozen discordance estimates. Claims of
effect removal require TOST and a 90% CI inside [-0.10, +0.10]; non-significance alone is invalid.

## Valid outcomes and authorization

The only valid outcomes are `SEMANTIC_PRIMARY_GO`, `INTEGRATION_INDEPENDENT_GO`,
`JOINT_CAUSAL_GO`, `NO-GO_ALTERNATIVE_MECHANISM`, and `INCONCLUSIVE`. A joint result is a valid GO,
not a failure to find one cause. Any GO authorizes at most an independently frozen backbone
replication (plus construct-validity analysis for joint causality). Model development and paper
writing remain false under every LLaVA-only outcome.

