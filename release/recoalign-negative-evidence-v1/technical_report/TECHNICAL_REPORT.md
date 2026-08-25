# ReCoAlign Research Closeout and Negative-Evidence Technical Report

> This technical report documents a preregistered sequence of supported, falsified and
> inconclusive results. It is not a submission manuscript or a validated method report.

## 1. Project scope

ReCoAlign examined whether structured relational evidence could supply a privileged reasoning
interface to a frozen vision-language model. The program is now closed. This report preserves what
was tested, what failed, the narrow result that remained supported, and why no successor method or
claim-bearing paper was authorized.

## 2. Original hypothesis

H001, the Structured Reasoning Interface Gap, predicted that a graph interface would outperform an
information-equivalent caption interface. EXP001 produced Graph < Caption and a preregistered
NO-GO. H001 was falsified and retired.

## 3. Experimental governance

The program used preregistration, frozen configurations, explicit integrity and decision gates,
seed-completeness requirements, artifact manifests, and GO/NO-GO/INCONCLUSIVE adjudication.
Measurement validity was kept separate from semantic validity. Failed and inconclusive studies
remain in the evidence record. Final candidate selection required every mandatory evidence,
identifiability, novelty, feasibility, and post-hoc-risk gate to pass.

## 4. Frozen model and runtime

Scientific inference used `liuhaotian/llava-v1.5-7b` at revision
`4481d270cc22fd5c4d1bb5df129622006ccd9234`. The original evidence manifest records Python 3.12.6,
PyTorch 2.7.1+cu126, Transformers 4.52.4, bitsandbytes 0.48.2, CUDA 12.6, float16 with NF4
quantization, and the legacy Transformers loader. The release contains provenance but no model
weights.

## 5. Experiment sequence

EXP001 tested graph versus information-controlled caption input. EXP002 tested correct supplied
relations against partial, corrupt, and random evidence. EXP003 tested an OOD claim but failed its
token-length integrity gate. EXP004 measured visual semantic availability. PIVOT_EXP_A evaluated a
semantic–structural integration account; PIVOT_EXP_A2 attempted causal separation. PIVOT_EXP_A3 and
A3R tested free-generation measurement contracts. PIVOT_EXP_A3P replaced the retired secondary
instrument with a primary conditional-likelihood instrument and evaluated M1/M2. The final pivot
gate assessed PH005, PH006, PH007, and STOP.

## 6. GO, NO-GO and INCONCLUSIVE outcomes

| Study | Outcome | Governed meaning |
| --- | --- | --- |
| EXP001 | NO-GO | Graph did not provide the predicted privileged interface. |
| EXP002 | GO | Correct supplied relations mattered within the frozen single-backbone scope. |
| EXP003 | INCONCLUSIVE | Token-length integrity failed; no OOD claim is permitted. |
| EXP004 | NO-GO | The required high visual-semantic-availability premise was unsupported. |
| PIVOT_EXP_A | NO-GO | Multiple mechanisms were supported; causal primacy was unresolved. |
| PIVOT_EXP_A2 | INCONCLUSIVE | Semantic-sufficiency and parsing-integrity gates failed. |
| PIVOT_EXP_A3 | RUNTIME_BLOCKED_PREINFERENCE | The full-field free-generation contract failed development smoke. |
| PIVOT_EXP_A3R | SECONDARY_CONTRACT_RUNTIME_FAILURE | 127/160 outputs were valid; 33 continued entity IDs; instrument retired. |
| PIVOT_EXP_A3P | SEMANTIC_MANIPULATION_FAILURE | 27,000/27,000 predictions were valid; M1 and M2 failed Gates C, D, E. |
| Final gate | STOP SELECTED | PH005, PH006, and PH007 each failed mandatory selection gates. |

## 7. Supported limited findings

EXP002 supports only supplied-relation sensitivity in the frozen LLaVA-1.5-7B configuration:
correct relational evidence performed better than partial, corrupt, or random supplied evidence.
PIVOT_EXP_A3P additionally supports the integrity of its primary prediction/scoring instrument.
Neither result establishes an internal scene graph, a structured interface gap, a general VLM
mechanism, or an effective ReCoAlign method.

## 8. Falsified claims

The privileged graph-interface claim, the proposed high-SAS premise, and both semantic rescue
manipulations were falsified under their registered gates. H001 and H004 are retired. M1 and M2 are
falsified. The project record does not convert those failures into a renamed success claim.

## 9. Failed measurement instruments

PIVOT_EXP_A3 failed before validation inference because its complete-field free-generation contract
did not survive development smoke. PIVOT_EXP_A3R's continuation contract yielded 127 valid outputs
and 33 entity-ID continuations in 160 development cases. The secondary free-generation instrument
was permanently retired. These are measurement-instrument failures, not evidence that the model
did or did not possess the targeted semantics.

## 10. Failed semantic manipulations

PIVOT_EXP_A3P produced all 27,000 preregistered primary predictions across five complete seeds. The
conditional-likelihood instrument passed its integrity gates. M1 and M2 nevertheless failed Gates
C, D, and E; therefore no semantic manipulation was selected and no method-development target was
authorized.

## 11. Final pivot assessment

PH005, PH006, and PH007 were assessed against the frozen evidence and contemporary adjacent
literature. Each failed at least one mandatory gate. `STOP` was selected under the registered
decision policy. No new preregistration, power analysis, experiment, or paper roadmap was created.

## 12. Limitations

The execution is single-backbone and checkpoint-specific. EXP003 and PIVOT_EXP_A2 are inconclusive,
not negative proofs. Low SAS does not imply semantic absence in every hidden layer. Failed
free-generation contracts do not resolve semantic capability. No independent replication or
cross-model generalization is represented.

## 13. Reproducibility

The repository preserves compressed predictions, metrics, decisions, protocols, figures,
provenance, and SHA-256 manifests. Integrity-only verification is available without checkpoints or
restricted datasets. The evidence-freeze tag anchors the scientific bytes; deterministic release
archives and `SHA256SUMS` anchor distributed assets.

## 14. Research integrity lessons

The closeout demonstrates why integrity gates precede interpretation, why instrument validity must
not be relabeled as construct validity, why a single positive sensitivity result must remain
scope-bounded, and why failed and inconclusive results require the same discoverability as GO
results. Explicit stop rules prevented repeated post-hoc pivots from becoming new claims.

## 15. Program closeout

The formal decision is `TERMINATE_CURRENT_PROGRAM`. Active method development, new experiments,
new preregistrations, real inference, independent-backbone replication, and claim-bearing paper
writing are not authorized. Non-claim technical reporting, release engineering, artifact-integrity
maintenance, and long-term archival are authorized.

```yaml
paper_writing_allowed: false
submission_ready: false
non_claim_technical_report_allowed: true
archive_and_release_allowed: true
```
