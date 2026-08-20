# ReCoAlign ablation protocol

All ablations inherit a registered stage configuration and change one causal factor:

| Config | Intervention | Interpretation |
| --- | --- | --- |
| `no_structure.yaml` | zero learned structure tokens and structural loss | necessity of the interface |
| `random_structure.yaml` | fresh random tokens, no structural loss | extra-token capacity control |
| `no_semantic_loss.yaml` | semantic weight = 0 | semantic preservation necessity |
| `no_reasoning_loss.yaml` | reasoning weight = 0 | reasoning-path alignment necessity |

The random-label sanity test is separate: it preserves architecture and optimization
while permuting object, attribute, relation, and composition training labels. It is
evaluated on the unchanged clean validation labels.

Original VLM, caption adapter, oracle graph prompt, and ReCoAlign comparisons must
pass `validate-training-fairness`. Vision backbone, LLM, dataset, split, and seed are
identical. Oracle graph prompting is marked as an upper bound and is never relabeled
as ReCoAlign inference.

Failed or neutral ablations remain reportable. Ablation results cannot be used to
retune the frozen evaluation split.
