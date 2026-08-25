# Statistical Protocol

All task accuracy and paired-difference intervals use 10,000-sample hierarchical seed-scene
percentile bootstrap with base seed 20260826. Five seeds are the first resampling level and scenes
within sampled seeds are the second. At least four of five seed means must meet each registered
task-level mean threshold.

Gate E uses paired oracle-minus-corrupted scene differences and Holm correction across four tasks.
Gate F uses paired neutral-minus-original scene differences and two one-sided equivalence tests at
margin ±0.05, with Holm correction on the maximum one-sided p-value. A non-significant difference
cannot establish equivalence. All four tasks must pass; aggregate performance cannot override a
failure.
