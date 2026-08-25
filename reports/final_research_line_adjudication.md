# Final ReCoAlign Research-Line Adjudication

## 1. Original claim

The original Structured Reasoning Interface Gap claimed that VLMs preserve sufficient visual
semantics but lack an accessible structured intermediate representation, and that learning this
interface would improve compositional and OOD reasoning.

This claim is **falsified**. It is not an active hypothesis or method premise.

## 2. Falsifying evidence

- EXP001: Graph underperformed information-equivalent Caption; decision `NO-GO`.
- EXP004: SAS was low; the required high-semantic/low-structured-access phenotype was absent;
  decision `NO-GO`.
- EXP003: the token-length integrity gate failed; it cannot support an OOD rescue.
- PIVOT_EXP_A: primitive behavior was low, relation contribution was positive, and serialization
  effects were stable; multiple explanations survived, so causal primacy was not identified;
  decision `NO-GO`.
- PIVOT_EXP_A2: semantic-sufficiency and parsing-integrity gates failed; `INCONCLUSIVE`.
- PIVOT_EXP_A3P: no valid semantic rescue manipulation was selected; M1/M2 failed C/D/E;
  `SEMANTIC_MANIPULATION_FAILURE`.

## 3. Supported limited findings

- EXP002 establishes that correct supplied relational evidence outperforms corrupted, partial, and
  random evidence in the frozen LLaVA-1.5-7B scope only.
- EXP001 establishes caption-over-graph behavior in its frozen scope.
- A3P establishes technical validity of the primary conditional-likelihood scorer and 27,000-row
  prediction inventory.

These findings do not establish an internal scene graph, semantic sufficiency, a universal VLM
mechanism, or method efficacy.

## 4. Failed construct attempts

- A2's oracle semantic scaffold did not satisfy semantic sufficiency and its parser failed on 137
  manipulation checks.
- A3's frozen full-field free-generation contract failed development smoke before inference.
- A3R's continuation contract produced 127/160 valid responses and 33 entity-ID continuations; the
  secondary instrument was retired before validation.
- A3P preserved a valid forced-choice measurement but falsified M1 and M2 as sufficient semantic
  manipulations.

No third free-generation repair, M3, new parser, prompt, or semantic scaffold is authorized.

## 5. What cannot be claimed

The evidence cannot support graph superiority, a Structured Reasoning Interface Gap, general
semantic absence, semantic comprehension from measurement validity, a dominant conditional
mechanism, a valid OOD claim, cross-model generality, ReCoAlign effectiveness, or any training,
adapter, token, loss, or alignment method claim.

## 6. Remaining candidate explanations

| Candidate | Gate-relevant assessment | Result |
| --- | --- | --- |
| PH005 Arbitrary Referential Binding Limitation | Direct support 2/5; distinctness 3/5; identifiability 4/5; feasibility 4/5; Q1 2/5; post-hoc risk 5/5; RDoF risk 2/5 | FAIL |
| PH006 Serialization-Conditioned Evidence Usability | Direct support 3/5; distinctness 2/5; identifiability 3/5; feasibility 4/5; Q1 2/5; post-hoc risk 4/5; RDoF risk 4/5 | FAIL |
| PH007 Cross-Modal Evidence Competition | Direct support 2/5; distinctness 2/5; identifiability 3/5; feasibility 4/5; Q1 2/5; post-hoc risk 4/5; RDoF risk 3/5 | FAIL |
| STOP | Zero scientific candidates pass all mandatory gates | SELECTED |

Concrete criterion-by-criterion reasons are recorded in
`research/final_pivot_candidate_matrix.yaml`.

## 7. Novelty audit

The primary-literature audit found:

- arbitrary-symbol mapping is directly studied by Symbol Tuning (EMNLP 2023), and instance-level
  VLM attribute misbinding is directly measured across seven LVLMs in 2026;
- VLM/LMM prompt structure and phrasing sensitivity is already benchmarked across many models and
  datasets by RobustPrompt/MVP and Promptception;
- controlled cross-modal source conflict, degradation, conditional-likelihood arbitration, and
  multimodal conflict are already studied across broad VLM families by Which Source Wins? and
  SIGNPOST-Bench.

Novelty status is `VERIFIED_NO_CANDIDATE_PASSES`. The complete audit is
`research/final_pivot_literature_audit.md`.

## 8. Stop-or-pivot decision

**TERMINATE_CURRENT_PROGRAM**

No candidate satisfies every mandatory selection gate. Selecting any one would require lowering
the evidence, distinctness, identifiability, Q1, or post-hoc-risk threshold. STOP is therefore
mandatory. No final-pivot preregistration or power analysis was created.

## 9. Authorization

```yaml
selected_candidate: STOP
new_experiment_allowed: false
new_preregistration_allowed: false
real_inference_allowed: false
independent_backbone_replication_allowed: false
model_development_allowed: false
paper_writing_allowed: false
claim_bearing_paper_allowed: false
maximum_additional_diagnostic_pivots: 0
archive_and_release_allowed: true
non_claim_technical_report_allowed: true
```

## 10. Resource release recommendation

Release is appropriate as a negative-results repository, reproducibility resource, internal
technical report, and research-evidence-governance example. The repository is not suitable, in its
current state, as an SCI Q1 claim-bearing paper. Historical method modules should remain intact and
marked inactive for provenance.

