# Paper Readiness Report After PIVOT_EXP_A2

## Decision

**NO-GO**

Do not begin claim-bearing paper writing and do not treat the current results as evidence that
ReCoAlign solves a Structured Reasoning Interface Gap.

## Current evidence state

| Required claim | Evidence outcome | Status |
|---|---|---|
| Structured Reasoning Interface Gap exists | EXP001 NO-GO; EXP002 GO; EXP003 INCONCLUSIVE | unsupported |
| Visual semantics are preserved | EXP004 SAS 0.1214; Type A semantic failure | falsified under current probe |
| ReCoAlign learns the missing interface | no real trained-interface result in this execution | missing |
| Improvement generalizes | EXP003 rejected by token-length integrity gate | missing |
| Pattern holds across VLMs | only LLaVA-1.5-7B executed | missing |
| Conditional mechanism is identified | PIVOT_EXP_A2 INCONCLUSIVE | unsupported |

The repository now contains a complete five-seed conditional-intervention asset with 5,300 real
LLaVA-1.5-7B predictions. PIVOT_EXP_A2 nevertheless remains `INCONCLUSIVE`: the oracle semantic
scaffold did not reach the preregistered sufficiency gate, and 137 manipulation-check responses
failed the frozen answer parser. This is a valid stopping result, not paper readiness.

## PIVOT_EXP_A2 adjudication

- Prediction inventory: 5,300/5,300; five seeds each contain 720 main, 180 reference, and 160
  manipulation-check cells.
- Matrix, token matching, corruption validity, freeze hashes, and artifact manifest: passed.
- Parser integrity: failed; 137/5,300 outputs were unparsed, all in manipulation checks.
- Semantic sufficiency: failed; oracle accuracy 0.7275, 95% CI [0.6725, 0.7800], below mean 0.90
  and CI-lower 0.85 thresholds.
- Formal outcome: `INCONCLUSIVE`.
- Authorization: independent replication false; model development false; paper writing false.

The observed E2 relation contrast (0.3156, 95% CI [0.2744, 0.3567]) and E3 JSON-minus-triples
contrast (0.0778, 95% CI [0.0333, 0.1289]) are retained as descriptive diagnostics only. They
cannot establish semantic-primary, integration-independent, or joint causality because the
upstream manipulation and integrity gates failed.

## Why EXP002 does not rescue the claim

EXP002 robustly shows that a correct externally supplied graph outperforms missing, random, wrong,
or opaque-label graphs. This demonstrates that LLaVA can use structural evidence and is sensitive
to its correctness. It does not isolate an internal perception-to-structure interface bottleneck:

- Caption still outperforms Graph in EXP001;
- semantic availability is low in EXP004 rather than high;
- no valid OOD retention result exists;
- no learned ReCoAlign interface was evaluated.

The defensible statement is “the model can exploit correct graph evidence,” not “the model has
semantics but lacks a structured interface.”

## Reviewer attack points

1. **Semantic-failure alternative is observed, not excluded.** EXP004 directly selects Type A.
2. **Graph prompting is not superior to caption prompting.** EXP001 effects are significantly
   negative under natural and token-matched controls.
3. **Structure sensitivity is external-input sensitivity.** EXP002 alone cannot identify an
   internal interface mechanism.
4. **OOD evidence is invalid under the registered controls.** EXP003 failed length matching.
5. **No cross-model replication exists.** A single LLaVA checkpoint cannot support a VLM-wide
   mechanism claim.
6. **No learned-method evidence exists.** The execution did not produce a trained ReCoAlign
   checkpoint or real hidden-context interface result.
7. **Hidden-state diagnosis is incomplete.** The language-side visual hidden representation is
   unavailable in the active adapter.

## Required next action

The conditional study has been tested and did not pass its identification gates. The only
scientifically authorized next step is a new construct-validity review and preregistration, not
automatic replication, benchmark expansion, method training, or paper writing:

1. audit why entity-binding oracle scaffolds fail the fixed-answer contract without reclassifying
   the retained 137 responses;
2. design a behavior-independent semantic manipulation that can actually satisfy the registered
   sufficiency threshold without leaking relations or answers;
3. preregister its evaluator and parsing contract before any new inference;
4. retain PIVOT_EXP_A2 unchanged as an inconclusive construct-validity result;
5. do not replicate across backbones, train, or promote ReCoAlign until a future manipulation passes
   its frozen sufficiency and integrity gates.

All current positive and negative evidence is retained under `reports/paper_evidence/`.
