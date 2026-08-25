# ReCoAlign Cross-Project Contamination Audit

## Repository identity

- Repository: `git@github.com:Yangjunjie-Lin/recoalign.git`
- Working branch: `codex/claim-evidence-execution`
- Audited local HEAD: `cf6e7fc24f1f2d36c252e7060e6eb4fdc53fd19d`
- Audit date: 2026-08-24
- Active domain: vision-language model construct validity, compositional reasoning, semantic
  availability, relational evidence, and answer-contract measurement.

The reported object `2788f141e276df1158098a0c3704319641d25c60` is not present in the local
ReCoAlign Git object database, is not reachable from any local ReCoAlign ref, and is not the object
ID of any ref advertised by `origin`. It therefore cannot be used as ReCoAlign provenance or
scientific evidence.

## Tracked-source term search

The required tracked-source search was executed before any amendment work:

```text
git grep -n -E "CheckCorr|SAFE_STOP|commitment divergence|finite semantic|Lean build|action alias"
```

It returned one lexical match: `README.md:64`, where ReCoAlign's synthetic-world ontology is
described as containing “finite semantic factors and typed relations.” The surrounding section is
the ReCoAlign repository map and contains no foreign algorithm, theorem-proving workflow, or
foreign execution instruction. This is a non-relevant lexical overlap, not cross-project
contamination. Relevant tracked-source matches: 0.

## Excluded context

The previously reported Prompt 17R material and its CheckCorr, theorem-proving, SAFE_STOP,
action-alias, and commitment-divergence concepts describe a different project. They do not concern
LLaVA-1.5-7B, PIVOT_EXP_A3, semantic manipulations M0/M1/M2, CV1/CV2/CV3, or ReCoAlign's frozen
answer contract. No such material was implemented, executed, or accepted as evidence here.

During A3R implementation, a project-consistency helper briefly embedded the excluded term names as
string literals for a self-scan. The final tracked-source hard gate detected this lexical
contamination. The helper and every foreign string literal were removed before finalization; no
foreign algorithm or workflow was implemented, and no scientific artifact or outcome was changed.

## Disposition

- Cross-project scientific evidence imported: no
- ReCoAlign tracked source restored during this audit: A3R consistency helper
- Foreign prompt executed: no
- PIVOT_EXP_A3 validation inference inspected or started: no
- Project identity hard gate: PASS
