# Paper Readiness Report After Claim-Evidence Execution

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

The repository has moved from “no real-model evidence” to “real-model evidence with a negative
scientific decision.” That is meaningful progress, but it is not paper readiness.

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

The frozen hypothesis has been tested and did not pass. The next step is a scientific review, not
automatic benchmark expansion or method training:

1. audit whether the EXP004 probe task and 24-sample design validly measure semantic availability;
2. separately audit the EXP003 graph/random-graph token mismatch without changing the retained
   failed run;
3. decide whether to revise or retire the high-SAS/low-StAS hypothesis before any new execution;
4. if a revised hypothesis is preregistered, replicate on at least one additional VLM;
5. do not train or promote ReCoAlign until the diagnosis supports the claimed mechanism.

All current positive and negative evidence is retained under `reports/paper_evidence/`.
