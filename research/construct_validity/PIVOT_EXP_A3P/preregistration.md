# PIVOT_EXP_A3P Preregistration

## Timing and independence

This primary-only study was established after two development-only response-instrument failures and
before any PIVOT_EXP_A3 or PIVOT_EXP_A3R held-out prediction, scientific metric, or candidate
outcome was observed. Its repair basis is measurement-instrument failure only. It does not overwrite
either parent study.

## Frozen inventory

The validation sample is exactly the 27,000 rows with `response_method == forced_choice` in the
frozen PIVOT_EXP_A3 inventory: five seeds, 100 scenes per seed, and 54 primary cells per scene.
Every scientific and execution field is protected. Only the administrative row `study_id` changes;
the parent trial key remains unchanged. No scene, image, legend, question, option, answer position,
scaffold, corruption, prompt, hash, or trial key may be regenerated or amended.

## Measurement

The sole response method is `conditional_log_likelihood_single_token_argmax_v1`. The four candidate
continuations are the registered IDs `1`, `2`, `3`, and `4`, appended to prompts ending exactly in
`FINAL_CHOICE=`. Every row must contain exactly four finite log likelihoods and one registered
argmax. Exact ties use numeric registered-ID order. The scoring function receives only prompt,
image, and registered option IDs; truth fields are joined only after scoring. There is no generated
response, output interpretation, synonym, entity alias, reconstruction, fallback, answer extraction,
or model update.

To remain within the 6 GB CUDA boundary without changing the measurement, execution is staged:
the frozen vision tower and projector first compute exact image-token features on CUDA; those
modules are then released before the same NF4 language model performs the frozen next-token
forward on CUDA. The cache is keyed by image and model SHA. No model parameter executes on CPU,
and a development scorer-drift gate requires exact reproduction of the parent M0, M1, and M2
primary smoke scores before freeze.

## Design and endpoints

The frozen constructs are CV1 mapping comprehension, CV2 neutral-image scaffold comprehension, and
CV3 original-image scene-truth sufficiency. M0 is descriptive only. M1 is the canonical entity table;
M2 is the visual object legend. Evidence is oracle or corrupted. Tasks are shape, color, object
identity, and entity–attribute binding.

Gate A requires valid primary measurement on all 27,000 rows and, for M1 and M2 separately, CV1
oracle and corrupted means at least 0.95 with 95% confidence lower bounds at least 0.90. Corrupted
CV1 is scored against the active declaration.

Gate C requires every task under neutral image plus oracle scaffold to have mean at least 0.95 and
95% lower bound at least 0.90. Gate D requires every task under original image plus oracle scaffold
to have mean at least 0.90 and lower bound at least 0.85. Gate E requires, for every original-image
task under scene-truth scoring, oracle minus corrupted accuracy at least 0.50, a 95% lower bound
strictly above zero, Holm-controlled positive separation, and a corrupted condition that would not
independently pass Gate D. Gate F requires paired TOST equivalence of neutral-oracle and
original-oracle accuracy within ±0.05 for every task, with Holm correction. Non-significance is not
evidence of equivalence. No aggregate endpoint may override a task failure.

The former secondary Gate B is deleted from candidate logic because its external-validity
instrument was retired as invalid. Its status is `RETIRED_BEFORE_VALIDATION`, it never participates
in a scientific decision, and its negative evidence remains reported.

## Candidate rule

M1 is selected if it passes A, C, D, E, and F. M2 is selected only if M1 fails and M2 passes all five
gates. If both pass, M1 is selected by the minimum-intervention principle. If both fail, there is no
combination, additional manipulation, or post-hoc prompt.

## Inference and multiplicity

Confidence intervals use a fixed-seed hierarchical seed-scene percentile bootstrap. Gate E and F
task families use Holm correction at alpha 0.05. Gate F is the most demanding power endpoint. No
seed, failed trial, or task may be deleted, and thresholds cannot change after freeze.

## Development and execution boundary

The 432-cell development smoke uses seed 20260830 and evaluates runtime and measurement mechanics
only. It cannot compute correctness, task performance, evidence-condition contrasts, or candidate
superiority. Held-out inference is prohibited until parent hashes, inventory inheritance, prospective
power, the full smoke, runtime, and a pushed pre-inference commit all pass.

Formal execution uses atomic append, trial-key resume, no overwrite, no duplicate, per-seed
checkpoints, a fixed environment, and no intermediate-result-driven decision. Formal adjudication is
prohibited until all 27,000 rows and all five seeds pass integrity.
