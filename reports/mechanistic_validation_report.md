# ReCoAlign mechanistic validation report

## Scientific status

**INCONCLUSIVE for real VLMs; implementation and toy controls pass.** The current
EXP004 artifact is an infrastructure ReferenceVLM run with
`claim_status: infrastructure_validation` and a semantic-failure classification.
It cannot establish a real-VLM interface gap. Therefore this report does not promote
toy ablations or fixture interventions to paper evidence.

## Questions and registered tests

| Question | Evidence required | Current status |
| --- | --- | --- |
| Does the structure interface matter? | A2 no-interface and I1 removal paired gains | toy suite available; real matrix pending |
| Does learning matter? | A3 random tokens, A4 fixed graph encoder | random-token toy control available; fixed real encoder pending |
| Does structure improve generalization? | EXP003 IID/OOD paired matrix | frozen protocol available; trained cells pending |
| Is the gain extra parameters? | P1 parameter-matched random control | exact-count control implemented; real comparison pending |
| Is it oracle information? | S1/S2/S3 supervision controls and graph-free inference | supervision registry implemented; real runs pending |

All registered ablations remain visible, including neutral and failed outcomes.

## Causal intervention contract

The model exposes `forward_with_structure_tokens` only for interventions. Remove,
shuffle, and cross-sample replacement preserve tensor shape and never accept a graph.
Each result records accuracy delta, changed prediction fraction, token L2 delta, seed,
and `oracle_graph_used: false`.

## Interpretation rule

GO requires consistent multi-seed effects for interface removal/random tokens,
representation probes showing compositional information, causal damage under
intervention, parameter-matched non-reproduction, and failure reduction. A toy or
ReferenceVLM result alone cannot satisfy any real-model criterion.
