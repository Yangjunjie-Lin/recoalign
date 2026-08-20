# Real VLM Evaluation Protocol

## Frozen evaluation

Each model is evaluated without training or parameter updates on the unchanged EXP001, EXP002, and
EXP003 generators, interventions, metrics, and decision thresholds. The model registry is merged
into the registered experiment configuration in memory; benchmark configuration is not overridden.

## Prompt control

Every model receives `configs/prompts/reasoning_default.yaml`. This protocol is byte-equivalent to
the Phase-1 prompt:

```text
{evidence}
Question: {question}
Answer with exactly one option: {choices}.
```

Backbone chat templates may add required image tokens and role markers outside this scientific
content. Their adapter identity is recorded. Caption and graph condition files inherit the same
protocol and explicitly forbid model-specific override. Generation is deterministic: temperature
zero, sampling disabled, one beam, and a fixed token limit.

## Answer evaluation

The shared evaluator applies exact match, Unicode/case/punctuation-normalized match, and unambiguous
multiple-choice extraction in that order. Optional reasoning text is retained, while a declared
`Final answer:` is extracted uniformly. Ambiguous responses containing multiple options are wrong.

Each prediction retains `sample_id`, `prediction`, `ground_truth`, `correct`, normalized values,
evaluation method, condition, split, seed, prompt version/hash, and input accounting.

## Statistics and evidence eligibility

Single-seed execution is diagnostic only. Default validation requires at least three seeds; critical
EXP001–EXP003 claims retain their five registered seeds, mean, standard deviation, 95% confidence
interval, and paired tests. ReferenceVLM is never claim-bearing. LLaVA-1.5 becomes eligible only for
a complete pinned multi-seed run. Unpinned model families remain INCONCLUSIVE.

## Failure handling

Failures are never dropped. Predictions are assigned to object, relation, multi-hop, or structure
misuse categories and written under `reports/failure_analysis/`. A graph error on a scene whose
caption answer is correct is specifically marked `structure_misuse` for subsequent diagnosis.
