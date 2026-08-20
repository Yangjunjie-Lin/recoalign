# ReCoAlign diagnostic architecture after the research pivot

The codebase retains its graph and reasoning interfaces because they reproduce the frozen
experiments. They are diagnostic instruments, not an asserted model architecture.

```text
frozen scene state
   ├── image
   ├── entity / attribute evidence
   ├── relation evidence
   └── fluent or triple serialization
                ↓
           frozen BaseVLM
                ↓
 representations + answer predictions
                ↓
 semantic / relation / binding probes
                +
 paired factorial effects and integrity gates
```

## Active design rules

1. Keep the VLM frozen during hypothesis revision.
2. Treat graphs, captions, corruptions, and oracle facts as interventions, not model claims.
3. Separate semantic correctness, relation correctness, and serialization format.
4. Match facts and token budgets where the registered estimand requires them.
5. Use scene-disjoint probes with random-label and capacity controls.
6. Do not implement a new loss or adapter before a replicated diagnostic GO.
7. Preserve all EXP001–EXP004 artifacts and their original protocols.

## Stable infrastructure

`src/recoalign/` remains authoritative for VLM lifecycle, controlled-world records, experiment
governance, checkpoint/dataset manifests, evaluation, provenance, and paper-evidence integrity.
Retrieval remains a capability-preservation boundary.

## Historical interfaces

`models/structure_encoder/`, `models/reasoning_interface/`, and `models/recoalign/` remain available
for reproduction and compatibility. Their presence is not evidence that a structured interface is
the correct future intervention.

## Next architecture decision

None. The next authorized artifact is a preregistered PIVOT_EXP_A protocol. Model architecture is
explicitly deferred.
