# PIVOT_EXP_A3P Primary Immutability Report

PIVOT_EXP_A3P inherits only the frozen forced-choice rows from PIVOT_EXP_A3. The
administrative `study_id` is the sole row-level change; trial keys remain the parent keys.

- Expected rows: 27,000
- Actual rows: 27,000
- Exact protected-field matches: 27,000
- Mismatches: 0
- Unique image assets verified: 2,501
- Parent inventory SHA-256: `bcd2e86e7a97bcd879d161b59eada31bbac73dac10d0d77f40db4a55e065d27d`
- Inherited inventory SHA-256: `7b81e65ddb6ec72b692b8ab09ec05d6d9dbcd587a2b40a6e5b77572f43416ca2`

The parent inventory does not materialize an `image_sha256` field in individual rows. That
absence is itself protected and matches in all 27,000 comparisons. Image bytes are independently
checked against the frozen parent visual-asset manifest. No scene, image, legend, question,
choice, answer mapping, scaffold, prompt, prompt hash, corruption, or trial key was regenerated or
rewritten.
