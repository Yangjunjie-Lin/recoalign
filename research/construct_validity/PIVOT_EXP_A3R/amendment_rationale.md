# PIVOT_EXP_A3R Amendment Rationale

## Pre-inference basis

PIVOT_EXP_A3 v1 stopped with `RUNTIME_BLOCKED_PREINFERENCE`. Its design preflight, CUDA/NF4 load,
primary conditional-likelihood path, determinism check, visual-legend path, text-table path, and
cache check passed. The sole development-runtime failure was a secondary generation whose raw
output was the ASCII digit `2` while v1 required the model itself to emit `FINAL_CHOICE=2`.
Validation inference had not started, no validation outcome was observed, and no scientific metric
was computed. The repair basis is therefore development-runtime-smoke-only.

## Measurement diagnosis

The output demonstrated a boundary mismatch between a prompt asking the model to render a complete
field and a decoder returning only its value. PIVOT_EXP_A3R prospectively freezes the field name and
delimiter inside the prompt and measures only the generated continuation. The prompt ends exactly
at `FINAL_CHOICE=` with no trailing space. The raw continuation is retained unchanged. A complete
surface form is mechanically reconstructed as frozen prompt suffix plus raw continuation.

This is not retrospective prefix injection or parser relaxation: the prefix exists before model
inference; both the raw continuation and reconstruction are stored; two independently frozen ASCII
grammars are checked; their choice IDs must agree; and parsing never receives ground truth or
correctness. The v1 output remains invalid under v1 and is not reinterpreted.

## Scope

The scientific question, CV1/CV2/CV3, M0/M1/M2, seeds, scenes, images, choices, correct mappings,
primary conditional-likelihood method, model revision, NF4 configuration, Gates A/C/D/E/F, power
target, statistical protocol, selection rule, and downstream prohibitions remain unchanged. Only
the secondary free-generation surface boundary changes.

