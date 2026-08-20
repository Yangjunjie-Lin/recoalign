# EXP004 — Structured Reasoning Interface Diagnosis

## Scientific Question

At which stage does a frozen VLM fail: visual semantic availability, structured representation
accessibility, or reasoning execution?

## Hypothesis

H004 predicts that visual semantics can be measurable in image representations while relational
structure is less accessible, and that an oracle graph improves reasoning when the image-only path
fails.

## Variables

Independent variables are representation location, probe target, interface condition, and model.
Dependent variables are semantic availability (SAS), structured accessibility (StAS), reasoning
execution (RES), graph reconstruction F1, latent relation-probe accuracy, structure consistency,
oracle graph gain, and corrupted-graph sensitivity.

## Controlled Factors

All probes use frozen backbones, the registered synthetic generator, scene-disjoint deterministic
splits, fixed probe seeds, fixed decoding, and no parameter updates. Oracle graphs are generated from
the scene record and are never treated as model predictions. Missing hidden-state APIs are recorded
as unavailable rather than imputed.

## Stage measurements

1. Visual semantic availability: deterministic ridge linear probes over image-encoder,
   projected-visual-token, and language-side visual-hidden representations for object, attribute,
   relation, and composition labels.
2. Structured representation accessibility: graph reconstruction node/relation F1 and graph edit
   distance when the adapter exposes reconstruction; latent relation probes and same-relation versus
   different-relation consistency otherwise.
3. Reasoning execution: paired image-only, oracle-scene-graph, and corrupted-graph accuracy using
   the unchanged Phase-1 evaluator and prompt protocol.

## Evaluation Metrics

SAS is the mean available Stage-1 probe accuracy. StAS is the mean of available graph-relation F1,
latent relation-probe accuracy, and normalized structure-consistency score. RES is oracle-graph
reasoning accuracy. The interface-gap score is SAS minus StAS; oracle gain is oracle-graph accuracy
minus image-only accuracy. Every scalar reports mean, standard deviation, and percentile-bootstrap
95% confidence intervals across five probe/evaluation seeds.

## Statistical Protocol

Probe and evaluation seeds are fixed in the configuration. Scene-level pairing is preserved for
Stage-3 contrasts. A five-seed paper run is required for a claim-bearing model; ReferenceVLM remains
infrastructure validation and cannot produce a scientific GO decision.

## GO / NO-GO Criteria

GO requires high SAS, lower StAS than SAS by the preregistered interface-gap threshold, positive
oracle graph gain, all required measurements available, and consistent results across eligible
models. Type-A semantic failure, Type-B structure-interface failure, and Type-C reasoning-execution
failure are mechanistic classifications, not post-hoc model improvements. Any unavailable hidden
representation or incomplete eligible-model run yields INCONCLUSIVE rather than an imputed score.
