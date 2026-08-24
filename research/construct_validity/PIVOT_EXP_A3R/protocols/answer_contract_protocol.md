# PIVOT_EXP_A3R Answer Contract Protocol

Primary scoring remains `conditional_log_likelihood_single_token_argmax_v1`. Every primary prompt
ends in `FINAL_CHOICE=` and the scorer evaluates exactly the single-token continuations `1`, `2`,
`3`, and `4`. Tie-breaking, score aggregation, and all choice IDs are unchanged. Secondary output
never contributes to primary accuracy.

For secondary trials, the frozen prompt ends with:

```text
Return one registered option by completing the final field.

FINAL_CHOICE=
```

There is no trailing space. The decoder generates only a continuation. The run stores three
separate fields: `prompt_completion_prefix`, the verbatim `raw_continuation`, and
`reconstructed_contract_output`. Raw continuation must match `^\s*([1-4])\s*$`; reconstruction
must match `^\s*FINAL_CHOICE\s*=\s*([1-4])\s*$`; and the captured IDs must be identical.

No Unicode normalization, word-number conversion, substring search, alias matching,
multiple-option acceptance, or reasoning-trace extraction is permitted. Empty output and any
prefixed, explanatory, duplicated, Unicode-digit, or multi-option continuation is invalid and is
retained.

