# PIVOT_EXP_A3R Preregistration

## Parent boundary and amendment timing

PIVOT_EXP_A3 v1 is frozen as `RUNTIME_BLOCKED_PREINFERENCE` after its development-only secondary
contract smoke returned raw output `2` where the v1 full-field grammar required
`FINAL_CHOICE=2`. The parent design preflight and GPU runtime passed, including its unchanged
primary forced-choice scorer. No validation prediction was generated, no validation outcome was
observed, and no scientific metric was computed. PIVOT_EXP_A3R is a new study, not an overwrite or
reinterpretation of v1. Its repair basis is development-runtime-smoke-only.

## Unchanged scientific design

The research question, CV1 answer-contract comprehension, CV2 semantic-scaffold comprehension,
CV3 scene-truth semantic rescue, M0 historical anchor, M1 canonical table, M2 visual legend,
oracle/corrupted manipulation, four semantic tasks, five validation seeds, 500 scenes, images,
choice ordering, answer mapping, 27,000 primary trials, model/checkpoint/revision, NF4/FP16 runtime,
Gates A/C/D/E/F, hierarchical statistics, power target, selection rule, and authorization policy
are identical to PIVOT_EXP_A3 v1. Validation inventory reuse is verified row-by-row and by SHA-256.

## Primary measurement

Primary remains `conditional_log_likelihood_single_token_argmax_v1`. Its prompt ends exactly in
`FINAL_CHOICE=` and scores the single-token candidate continuations `1`, `2`, `3`, and `4`.
Exactly four finite scores are required. Numeric registered order resolves ties. Constrained
decoding, fallback parsing, alternative IDs, score aggregation changes, and use of secondary output
for primary accuracy are prohibited.

## Prospectively amended secondary measurement

The secondary prompt ends exactly in the ASCII prefix `FINAL_CHOICE=` with no trailing space. These
three objects are distinct and separately recorded:

1. **Prompt prefix:** the already-present frozen suffix `FINAL_CHOICE=`.
2. **Raw generated continuation:** the model's verbatim continuation, such as `2`.
3. **Reconstructed surface contract:** the mechanical concatenation, such as
   `FINAL_CHOICE=2`.

The raw continuation grammar is `^\s*([1-4])\s*$`. The reconstructed grammar is
`^\s*FINAL_CHOICE\s*=\s*([1-4])\s*$`. Both must pass and capture the same ID. Parsing has no access
to ground truth and is independent of response correctness. The prefix is frozen in each prompt
before inference, so reconstruction is not post-hoc prefix injection. v1's old output is never
rescored under v3.

## Trials and inventory

The planned total remains 36,000 predictions: 27,000 primary forced-choice and 9,000 secondary
continuation-generation trials, with 7,200 trials per validation seed. Every protected primary
field—trial key, prompt SHA, image SHA, choices, and correct-choice mapping—must match v1 in all
27,000 rows. Every secondary prompt is rehashed and its key is versioned with
`response_contract=answer-contract-v3-continuation`. No A3 v1 prediction may be imported.

## Development smoke

Before validation, seed `20260830` alone is used for eight frozen development scenes. The smoke
covers M1/M2 × oracle × neutral/original for CV1, shape, color, object identity, and
entity–attribute binding. It tests runtime validity, both parsers, deterministic repeat, CUDA/NF4,
OOM absence, and no CPU fallback. It computes no development accuracy and cannot compare candidate
performance. Required rates are 1.0 for raw parse, reconstructed parse, and deterministic repeat,
with zero unparsed rows. Failure yields `SECONDARY_CONTRACT_RUNTIME_FAILURE` and blocks validation.

## Registered gates and decision

Gate B requires both raw-continuation and reconstructed-contract parse rates at least 0.99 and each
Wilson 95% lower bound at least 0.98; thresholds are unchanged in strength. All other gates and the
task-specific, five-seed statistical protocol are identical to v1. Formal outcomes are limited to
`CONSTRUCT_VALIDITY_GO_TEXT`, `CONSTRUCT_VALIDITY_GO_VISUAL_LEGEND`,
`ANSWER_CONTRACT_FAILURE`, `SEMANTIC_MANIPULATION_FAILURE`,
`IMAGE_INTERFERENCE_DIAGNOSIS`, `NO-GO_CONSTRUCT`, and `INCONCLUSIVE`.

Only either GO outcome authorizes PIVOT_EXP_A4 preregistration. Independent-backbone replication,
model development, and paper writing remain prohibited for every outcome.

