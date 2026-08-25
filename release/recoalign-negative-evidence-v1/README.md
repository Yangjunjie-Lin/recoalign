# ReCoAlign Negative-Evidence and Research-Closeout Resource

ReCoAlign is a closed negative-evidence and reproducibility resource.
It is not a validated method release and does not support a
Structured Reasoning Interface Gap claim.

The scientific record is frozen at commit
`a80882071a6cf17c275453319d78d879c1546e3a` and annotated tag
`recoalign-evidence-freeze-2026-08-25`. The formal program decision is
`TERMINATE_CURRENT_PROGRAM`. Failed and inconclusive studies are first-class release artifacts,
not omissions.

## Authorized uses

This resource may be used to:

- inspect the complete hypothesis lifecycle;
- verify frozen metrics, predictions, decisions, and artifact hashes;
- study preregistered GO/NO-GO/INCONCLUSIVE research governance;
- reproduce artifact-integrity checks without model inference;
- reuse the synthetic-world and evaluation infrastructure under the repository license;
- analyze negative results and failed measurement instruments; and
- cite the limited finding that supplied correct relational evidence affected results in the
  frozen LLaVA-1.5-7B setting.

## Claims not supported by this release

This resource does not establish that:

- ReCoAlign is an effective method;
- graph input is superior to information-equivalent caption input;
- visual semantics are sufficiently preserved;
- a structured reasoning interface gap exists;
- semantic rescue succeeded;
- out-of-distribution generalization was demonstrated;
- the findings generalize to all vision-language models; or
- a claim-bearing paper is ready for SCI Q1 or any other venue.

## Starting points

- [`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md): equal-visibility study inventory.
- [`HYPOTHESIS_LIFECYCLE.md`](HYPOTHESIS_LIFECYCLE.md): hypothesis retirement record.
- [`CLAIM_BOUNDARY.md`](CLAIM_BOUNDARY.md): permissible and prohibited interpretations.
- [`technical_report/TECHNICAL_REPORT.md`](technical_report/TECHNICAL_REPORT.md): non-claim
  closeout report.
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md): integrity-only reproduction procedure.
- [`RELEASE_MANIFEST.yaml`](RELEASE_MANIFEST.yaml): machine-readable provenance and hashes.

## Integrity-only validation

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
python -m recoalign validate-research
python -m recoalign validate-paper-package
```

These commands validate frozen records and software behavior. They do not authorize or perform
new scientific inference.
