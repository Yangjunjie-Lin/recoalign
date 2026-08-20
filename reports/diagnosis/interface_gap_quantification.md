# EXP004 Interface-Gap Quantification Report

EXP004 defines:

- `SAS`: mean Stage-1 object, attribute, relation, and composition probe accuracy;
- `StAS`: mean available graph-relation F1, latent relation-probe accuracy, and normalized
  structure-consistency score;
- `RES`: oracle-scene-graph reasoning accuracy;
- `interface_gap`: `SAS - StAS`;
- `oracle_graph_gain`: oracle-graph accuracy minus image-only accuracy.

Every completed seed records mean, standard deviation, and percentile-bootstrap 95% confidence
intervals. Hidden representations and graph reconstruction are marked unavailable when the adapter
does not expose them. No unavailable measurement is imputed.

The current ReferenceVLM bundle is `INCONCLUSIVE` by governance policy and must not be interpreted as
a real-VLM mechanism result.
