# Mechanistic analysis protocol

## Ablation families

The registry in `configs/ablations/mechanistic_registry.yaml` covers architecture,
loss, supervision, capacity, and causal intervention families. Each row declares a
hypothesis, one variable change, expected outcome, interpretation, and minimum seed
count. The registry contains 14 explicit rows; no unnamed ablation is reportable.

## Supervision controls

- S1 uses full object/attribute/relation/composition labels as training targets.
- S2 reduces structural labels to a preregistered weak subset.
- S3 removes graph labels and retains QA reasoning supervision.

None of these labels are forwarded as inference tokens. All supervision conditions
must use the same images, questions, optimizer budget, prompt, decoding, and seeds.

## Capacity control

`parameter_matched_control` creates a random projector plus inert parameters with
the exact ReCoAlign parameter count. `parameter_audit` records total, trainable,
non-trainable, and per-module counts. Equal parameter count is necessary but does
not substitute for matched compute or training budget, which remain locked in the
experiment configuration.

## Representation evidence

Structure tokens are probed for object, attribute, relation, and composition labels
with the scene-disjoint linear-probe protocol inherited from EXP004. Similarity,
connected-component clustering, and reasoning attention summaries are saved as
diagnostic evidence. Attention is descriptive; causal intervention is the required
attribution test.

## Reporting

Use `recoalign validate-mechanistic-registry` and `recoalign run-mechanistic-toy`.
The toy command writes `mechanistic_suite.json` and labels it
`toy_mechanistic_validation_only`. Real runs must consume the frozen comprehensive
matrix and retain all failed/neutral cells.
