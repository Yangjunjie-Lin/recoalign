# A2 — Evidence Factorization

## Scientific Question

How do object facts, bound relation facts, and their union change frozen LLaVA compositional
decisions relative to the same image alone?

## Hypothesis

H-B predicts a reproducible bottleneck in using relational evidence or combining it with object
evidence, conditional on the A1 primitive-semantic classification.

## Variables

- C0 `image_only`: no textual semantic facts.
- C1 `object_evidence`: object/attribute inventory only.
- C2 `relation_evidence`: query-role aliases and declared relation bindings only.
- C3 `complete_evidence`: exact union of C1 object facts and C2 relation facts.
- Dependent variable: accuracy on the unchanged compositional question.

## Controlled Factors

Image, question, choices, model, decoding, and underlying scene stay fixed. Query-role aliases
(`question subject`, `answer object`, `reference object`, `intermediate object`) prevent relation
evidence from leaking shape/color answers. The four prompts are padded with repeated punctuation-only
`.` units to the same LLaVA token budget within one token. Natural and matched token counts and the
number of padding units are retained per prediction.

## Evaluation Metrics

- object contribution: C1 − C0;
- relation contribution: C2 − C0;
- complete contribution: C3 − C0;
- integration surplus: C3 − max(C1, C2);
- additive synergy: C3 − C1 − C2 + C0;
- condition accuracy by question type and hop depth.

## Statistical Protocol

Use paired scene-level bootstrap intervals within seed and five-seed summaries. A stable effect has
mean gain at least 0.05, lower 95% confidence bound above zero, and positive direction in at least
80% of final seeds.

## GO / NO-GO Criteria

A2 identifies a usable factor only when its registered contribution is stable. Missing or negative
effects remain evidence; they are not repaired by changing aliases or prompts after inference.
Mechanism classification combines A2 with A1 and A3 and is NO-GO if two primary explanations remain
equally compatible.
