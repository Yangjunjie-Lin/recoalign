# EXP001 — Graph vs Text Controlled Validation

## Scientific Question

Does a structured representation improve reasoning beyond a textual description containing exactly
the same declared scene facts? H001 predicts a positive paired `scene_graph - caption` accuracy
effect. The claim concerns representation structure, not Graph vs Image and not additional facts.

## Hypothesis

H001, Structured Reasoning Interface Gap: if structure is an independently useful reasoning
interface, scene-graph evidence will outperform information-equivalent caption evidence.

## Variables

The independent variable is evidence representation (C0–C3), crossed with natural/token-matched
length setting. Dependent variables are accuracy, paired graph-minus-caption accuracy, depth/task
accuracy, error class, and token efficiency.

## Controlled Factors

- C0 `image_only`: visual baseline.
- C1 `object_list`: image plus the complete object/attribute inventory, without relations.
- C2 `caption`: image plus a deterministic grammatical verbalization of every object, attribute,
  and relation.
- C3 `scene_graph`: image plus the identical canonical fact set serialized as typed nodes/edges.

Each C2/C3 pair shares scene ID, image, question, answer, object facts, relation facts, and canonical
fact SHA-256. A non-equal hash is an integrity failure, not an analyzable observation.

## Length settings

- `natural`: retain natural serialization lengths and record model-tokenizer counts.
- `token_matched`: append content-free `<pad>` sentinels to the shorter C2/C3 evidence view until
  full prompt token counts differ by at most one token. The pad count and residual delta are stored.

Token matching never adds objects, attributes, or relations. Semantic units are counted separately
as objects + declared attributes + relations.

## Evaluation Metrics

The generator deterministically balances object, relation, attribute, and multi-hop questions.
Accuracy is reported overall, by task, and at depths 1/2/3/4. Errors are assigned to `object_error`,
`relation_error`, or `compositional_error`. Token efficiency is accuracy per 1,000 full input tokens.

The primary estimate is a paired graph-minus-caption accuracy difference on identical samples.
## Statistical Protocol

Five committed seeds are summarized with mean, sample standard deviation, 95% percentile-bootstrap
CI, positive-seed fraction, and a preregistered one-sided paired t-test against zero.

## Required ablations

- Graph order: deterministically shuffle relation-triplet order.
- Text structure: shuffle the same atomic object/attribute/relation tokens into unordered text.
- Graph serialization: compare canonical node/edge text, compact tuples, and canonical JSON.

These are reported symmetrically; no best-only ablation selection is allowed.

## GO / NO-GO Criteria

GO requires all of the following across five seeds: natural graph-over-caption gain at least 0.05;
token-matched gain at least 0.05; positive 2-hop and 3-hop token-matched effects; positive preliminary
composition-OOD token-matched effect; lower 95% CI above zero; one-sided p <= 0.05; and at least 80%
positive seed effects for every registered gate. Eligible frozen target-model evidence is mandatory.

An eligible complete run that fails any gate is NO-GO. Missing/ineligible evidence is INCONCLUSIVE.
ReferenceVLM is explicitly registered as infrastructure-only and therefore cannot produce GO or
NO-GO about H001 even if its configured synthetic accuracies numerically pass or fail the gates.

## Reproduction

```bash
python -m recoalign run-graph-vs-text --config configs/graph_vs_text.yaml --output outputs/EXP001
```

The command refuses to overwrite a non-empty output directory.

`configs/exp001_llava15.yaml` is the stage-2 local-checkpoint template. It must replace the reference
config in the registry through a reviewed preregistration amendment before its outputs become
eligible scientific evidence.
