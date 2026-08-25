# PIVOT_EXP_A3R Answer Contract Amendment Report

The v1 development output showed a generation-boundary mismatch: after a prompt requested a full
field, the decoder supplied only the field value. A3R therefore prospectively placed the fixed
ASCII field prefix `FINAL_CHOICE=` at the end of every secondary prompt and froze generation as a
continuation measurement before any validation prediction existed.

A3R stores the prompt completion prefix, the verbatim raw continuation, and their mechanical
concatenation as separate fields. The raw grammar accepts only one ASCII option digit with optional
ASCII whitespace. The reconstructed grammar independently accepts only the complete registered
field, and both captured IDs must agree. Parsing does not receive ground truth or response
correctness.

This is not retrospective parser relaxation. The prefix is already present at inference time; raw
output is never rewritten; v1 remains invalid under v1; and Unicode conversion, aliases, substring
search, multi-option extraction, word-number conversion, and reasoning-trace extraction remain
prohibited. Primary conditional-likelihood scoring is unchanged and never consumes secondary
output.
