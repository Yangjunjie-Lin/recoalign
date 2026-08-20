# Frozen research hypothesis

The machine-readable source of truth is [`research/frozen_hypothesis.yaml`](../research/frozen_hypothesis.yaml).

## Frozen statement

Vision-language models can preserve useful visual semantics while lacking a sufficiently
accessible structured intermediate representation between visual perception and language
reasoning; learning that interface can improve compositional reasoning and unseen-composition
generalization.

## Supporting claims

- Visual semantic availability can be measured separately from structured accessibility.
- Correct structure can improve reasoning beyond information- and token-controlled text.
- The effect must survive composition-disjoint OOD evaluation.
- ReCoAlign must learn the interface and must not require oracle graph input at inference.

## Falsification gates

The hypothesis is not supported if graph structure provides no controlled advantage, if
random/corrupted controls perform equally well, if the advantage disappears on OOD composition,
if oracle structure does not help where an interface diagnosis predicts it should, or if
parameter-matched/random-token controls explain the gain.

## Status

`frozen`. Current implementation and ReferenceVLM results are infrastructure evidence only;
the scientific submission decision remains `NO-GO` until the registered real-VLM evidence is
complete.
