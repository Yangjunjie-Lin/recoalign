# PIVOT_EXP_A3R Statistical Protocol

This protocol is identical to PIVOT_EXP_A3 v1 except for the prospectively amended secondary
surface measurement. Gate B separately requires raw-continuation parse rate at least 0.99 with
Wilson 95% lower bound at least 0.98, and reconstructed-contract parse rate at least 0.99 with the
same lower-bound threshold. The two parsed choice IDs must agree. Neither threshold is reduced.

All accuracy and paired-difference intervals use 10,000 hierarchical percentile bootstrap samples
with fixed seed `20260826`. Each replicate samples five seeds with replacement, then paired scenes
with replacement within sampled seed. Pairing is preserved across oracle/corrupted,
neutral/original, and response methods.

Gate E uses task-specific paired superiority tests with Holm correction over four tasks. Gate F
uses paired TOST at margin ±0.05; the maximum of the two one-sided p-values is Holm-corrected across
the four tasks. Both one-sided hypotheses must reject after correction, and the corresponding
equivalence interval must lie strictly within the margin. Failure to reject a difference is never
treated as equivalence.

All five seeds and every registered trial remain in analysis. Primary endpoints are answer validity,
shape, color, object identity, and entity binding—not total accuracy. Any task failure blocks a GO.
