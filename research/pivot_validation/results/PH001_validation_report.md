# PH001 Validation Report

## Decision

**NO-GO — MULTIPLE SUPPORTED MECHANISMS; CAUSAL PRIMACY UNRESOLVED**

PIVOT_EXP_A completed all five frozen seeds and 2,000 predictions with every integrity gate
passing. The evidence supports both primitive-semantic limitation and format-dependent use of
relational evidence. It does not identify one primary cause of compositional failure, so the
registered multiple-explanations rule blocks model development and paper writing.

## A1 Primitive semantic availability

| Task | Accuracy | 95% CI | Classification |
| --- | ---: | --- | --- |
| object shape | 0.500 | [0.400, 0.600] | LOW |
| attribute color | 0.580 | [0.545, 0.620] | LOW |
| direct relation | 0.385 | [0.375, 0.395] | LOW |

All three behavioral primitive dimensions fail the registered high-availability gate. This
supports H-A in the frozen LLaVA-1.5 scope; it does not prove that every hidden layer lacks the
corresponding information.

## A2 Evidence factorization

| Effect | Mean | 95% CI | Stable |
| --- | ---: | --- | ---: |
| object contribution | +0.025 | [-0.035, 0.085] | No |
| relation contribution | +0.185 | [0.140, 0.225] | Yes |
| complete contribution | +0.415 | [0.365, 0.475] | Yes |
| integration surplus | +0.230 | [0.175, 0.285] | Yes |
| additive synergy | +0.205 | [0.150, 0.255] | Yes |

Relation evidence and its combination with object evidence produce reproducible gains. This is
evidence that relational content and its joint use matter; it is not evidence that graph is a
privileged interface.

## A3 Representation format

| Condition | Accuracy | 95% CI |
| --- | ---: | --- |
| natural language | 0.605 | [0.585, 0.625] |
| triples | 0.540 | [0.515, 0.560] |
| JSON | 0.725 | [0.705, 0.745] |

All token-matched pairwise comparisons pass the registered format-sensitivity gate:

| Pair | Mean difference | 95% CI |
| --- | ---: | --- |
| natural language − JSON | -0.120 | [-0.150, -0.090] |
| natural language − triples | +0.065 | [0.040, 0.085] |
| triples − JSON | -0.185 | [-0.195, -0.175] |

This supports or refines H-B in scope: identical relation facts are not consumed invariantly across
formats. It does not establish a trainable remedy.

## Scientific adjudication

The frozen automated classifier returned `GO / PRIMITIVE_SEMANTIC_REPRESENTATION_LIMITATION` because
its primitive-limitation branch precedes its format-sensitivity branch. That branch ordering masks
simultaneous support for H-A and H-B. The automated output remains in raw `metrics.json`; the formal
decision is overridden by the preregistered rule that multiple compatible primary mechanisms yield
NO-GO.

## Updated research direction

PH001 is supported as a reproducible joint phenotype, but causal primacy remains unresolved. A
separate preregistered study would be required to condition relational-format effects on adequate
primitive availability. No adapter, structure token, new loss, benchmark expansion, paper writing,
or ReCoAlign v2 work is authorized by these results.
