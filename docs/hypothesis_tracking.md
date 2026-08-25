# Hypothesis lifecycle closeout

The active ReCoAlign hypothesis chain ended at `PIVOT_EXP_A3P`. Failed and inconclusive results
remain visible; none has been converted to a weaker post-hoc success claim.

| ID | Hypothesis or construct | Frozen evidence | Lifecycle status |
| --- | --- | --- | --- |
| H001 | Structured Reasoning Interface Gap | EXP001 NO-GO; EXP004 NO-GO | **falsified and retired** |
| H002 | Correct relational evidence sensitivity | EXP002 GO | **supported_in_frozen_LLaVA_scope_only** |
| H003 | OOD Generalization claim | EXP003 integrity failure | **inconclusive and retired** |
| H004 | Interface Diagnosis | EXP004 NO-GO | **falsified and retired** |
| PH001 | Joint semantic–structural phenotype as causal explanation | PIVOT_EXP_A NO-GO; A2 INCONCLUSIVE | **unresolved_and_not_identified; retired** |
| M1 | Canonical entity table provides sufficient semantic rescue | A3P Gates C/D/E fail | **falsified and retired** |
| M2 | Visual object legend provides sufficient semantic rescue | A3P Gates C/D/E fail | **falsified and retired** |

H002 is deliberately not a general VLM conclusion. It says only that correct externally supplied
relations affected the frozen LLaVA-1.5-7B protocol. It does not establish an internal scene graph,
a structured-interface gap, or cross-model generality.

The machine-readable closeout is in `research/current_line_closeout.yaml`; detailed retirement
records are in `research/rejected_or_retired_hypotheses/`.

## Final candidate gate

PH005, PH006, and PH007 were candidates, not promoted hypotheses. None passed all required evidence,
distinctness, identifiability, feasibility, novelty, Q1-potential, post-hoc-risk, and
researcher-degrees-of-freedom gates. `STOP` was selected. No PH008/PH009 chain is authorized.
