# Hypothesis tracking

The hypothesis registry is the source of truth for the scientific claim graph. Each entry requires
an ID, description, motivation, directional prediction, falsification statement, required experiment,
and lifecycle status.

## Initial Phase-1 hypotheses

| ID | Hypothesis | Required experiment | Falsifying observation | Status |
|---|---|---|---|---|
| H001 | Structured Reasoning Interface Gap | EXP001 Graph vs Text | Information-controlled graph fails any preregistered graph-over-text gate | testing |
| H002 | Structural Necessity | EXP002 Full vs Partial vs Random Graph | Full graph fails either registered negative-control comparison | testing |
| H003 | Compositional Generalization | EXP003 OOD Composition | OOD effect or split-integrity gate fails | testing |

The registry uses these statuses:

- `proposed`: scientifically stated but not yet bound to an executable protocol;
- `testing`: registered protocol exists and evidence collection may proceed;
- `supported`: the preregistered experiment received GO and independent review accepted provenance;
- `falsified`: the preregistered experiment received NO-GO under valid provenance;
- `retired`: superseded for a documented reason, never silently deleted.

GO is evidence under a bounded protocol, not proof that the hypothesis is universally true. NO-GO
must be retained because it constrains the research direction. A new experiment must reference an
existing hypothesis; a genuinely different claim requires a new hypothesis ID.

Run `recoalign validate-research` after every registry edit. Validation rejects duplicate IDs,
orphan experiments, broken bidirectional links, missing protocols/configs/manifests, configuration
condition drift, unregistered dataset/model versions, insufficient seeds, enabled training, or a
protocol missing any mandatory scientific section.
