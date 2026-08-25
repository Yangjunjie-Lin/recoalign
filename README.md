# ReCoAlign — closed research line and frozen negative-evidence resource

ReCoAlign is scientifically closed as of `PIVOT_EXP_A3P`.

```text
Program decision: TERMINATE_CURRENT_PROGRAM
Current method line: RETIRED
New experiments: not authorized
Model development: not authorized
Claim-bearing paper writing: not authorized
Archive and release: authorized
```

The repository preserves a complete hypothesis lifecycle rather than a success narrative. The
original **Structured Reasoning Interface Gap** was falsified. Its successor,
**Semantic–Structural Integration Failure**, remained unresolved and was not identified as a
single causal mechanism. M1 and M2 both failed the preregistered semantic-rescue gates, so no
semantic manipulation and no method target were selected.

## Frozen evidence through PIVOT_EXP_A3P

| Study | Frozen result | Governed interpretation |
| --- | --- | --- |
| EXP001 | Graph < Caption; NO-GO | Graph is not a privileged interface over information-equivalent text. |
| EXP002 | Correct > corrupt/partial/random; GO | Correct supplied relations matter in the frozen LLaVA-1.5 scope only. |
| EXP003 | INCONCLUSIVE | Token-length integrity failed; no OOD claim is valid. |
| EXP004 | Low SAS; NO-GO | High semantic availability was not supported; no all-layer absence claim follows. |
| PIVOT_EXP_A | NO-GO | Multiple mechanisms were supported; causal primacy was unresolved. |
| PIVOT_EXP_A2 | INCONCLUSIVE | Semantic sufficiency and parsing integrity failed. |
| PIVOT_EXP_A3 | RUNTIME_BLOCKED_PREINFERENCE | The full-field free-generation contract failed development smoke. |
| PIVOT_EXP_A3R | SECONDARY_CONTRACT_RUNTIME_FAILURE | 127/160 valid; 33 entity-ID continuations; instrument retired. |
| PIVOT_EXP_A3P | SEMANTIC_MANIPULATION_FAILURE | 27,000/27,000 valid forced-choice predictions; M1/M2 failed Gates C, D, E. |

PIVOT_EXP_A3P measurement integrity passed for the scorer and prediction inventory only. It did
not verify semantic comprehension. Its frozen assets are under
[`research/construct_validity/PIVOT_EXP_A3P/results/`](research/construct_validity/PIVOT_EXP_A3P/results/).

## Final pivot gate

PH005 Arbitrary Referential Binding Limitation, PH006 Serialization-Conditioned Evidence
Usability, and PH007 Cross-Modal Evidence Competition were evaluated against fixed evidence,
identifiability, novelty, feasibility, and post-hoc-risk gates. None passed every gate. `STOP` was
therefore selected. No final-pivot preregistration or power analysis was created.

The adjudication is in
[`reports/final_research_line_adjudication.md`](reports/final_research_line_adjudication.md), the
literature audit in
[`research/final_pivot_literature_audit.md`](research/final_pivot_literature_audit.md), and the
machine-readable claim state in [`docs/evidence_map.yaml`](docs/evidence_map.yaml).

## Research boundary

- Frozen predictions, metrics, figures, decisions, and manifests are immutable.
- EXP002 supports relation-evidence sensitivity only in the frozen LLaVA-1.5-7B setting.
- Free-generation instruments are retired; no prompt or parser repair is authorized.
- Training, reasoning-interface, alignment, comprehensive-matrix, and ablation modules are retained
  for provenance but are scientifically inactive.
- Retrieval is a capability-preservation control, not the new scientific contribution; benchmark
  infrastructure is likewise provenance only.
- A future independent decision would be required to authorize any experiment, backbone
  replication, model work, or claim-bearing manuscript.

See [`INACTIVE_RESEARCH_BOUNDARY.md`](INACTIVE_RESEARCH_BOUNDARY.md),
[`docs/research_identity.md`](docs/research_identity.md), and
[`research/current_line_closeout.yaml`](research/current_line_closeout.yaml).

## Repository use

The authorized use is provenance inspection, reproducibility checking, negative-results release,
internal technical reporting, and research-governance study. Historical commands remain available
to validate already frozen records; they are not a roadmap and do not authorize inference.

```bash
python -m recoalign validate-research
python -m recoalign validate-paper-package
pytest
ruff check .
```

## Repository map

```text
docs/                         evidence map, claim boundary, lifecycle, and closed roadmap
research/                     frozen protocols, results, closeout, literature audit, archives
reports/                      readiness and final adjudication reports
reports/paper_evidence/       immutable EXP001–EXP004 claim-evidence package
training/                     inactive historical implementation
models/reasoning_interface/   inactive historical implementation
models/alignment/             inactive historical implementation
configs/training/             inactive historical configs
configs/benchmarks/           comprehensive matrix retained but inactive
configs/ablations/            inactive historical configs
src/recoalign/                validation and provenance infrastructure
tests/                        governance and engineering regression checks
```

No “train ReCoAlign,” “run comprehensive benchmark,” or “paper submission” milestone is pending.
Those former roadmap entries are retired.
