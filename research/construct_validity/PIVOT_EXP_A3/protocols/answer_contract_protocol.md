# Answer Contract Protocol

Primary scoring appends each registered digit directly after the exact prompt suffix
`FINAL_CHOICE=` and reads its frozen next-token conditional log likelihood. The scorer accepts no
answer argument. All four finite scores and the selected argmax are retained. Primary validity is
the fraction of trials producing this complete score vector and must equal 1.00. CV1 accuracy is
a separate part of Gate A: option mapping must reach 0.95 mean with 95% CI lower bound 0.90 under
oracle and corrupted declarations separately. A complete score vector is not by itself evidence
that the model understood the mapping.

Secondary generation receives no completed prefix and must emit the full one-line field. Parsing is
performed once by the frozen v2 parser. Invalid rows remain in the denominator and prediction
artifact. Parser validation covers exact, whitespace, case, sentences, multiple choices, invalid
choices, aliases, reasoning, truncation, Unicode punctuation/digits, duplicate fields, and option
substrings.
