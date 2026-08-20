# ReCoAlign optimization protocol

## Objectives

Only the three method objectives are optimized:

1. Semantic preservation aligns pooled visual and structure representations.
2. Structural consistency supervises object, attribute, relation, and composition
   prediction heads. Graph annotations enter here as labels, never as input tokens.
3. Reasoning alignment applies answer cross-entropy to the adapted reasoning context.

Stage configurations explicitly set all three weights, including zero weights. A
zero-weight ablation does not create a fourth loss.

## Optimizer and scheduler

The optimizer factory supports AdamW, Adam, and SGD over trainable parameters only.
Schedulers support constant, linear, and cosine decay with optional warmup. Invalid
learning rates, warmup periods, or all-zero loss weights fail before training.

Gradient clipping is configured per stage. Checkpoints retain optimizer, scheduler,
epoch, global step, resolved config, dataset manifest, metrics, Git metadata, and
Python/NumPy/Torch RNG state for exact continuation.

## Freeze policies

- `interface_only`: train ReCoAlign interface, alignment, probe, and reasoning heads.
- `interface_and_projector`: additionally train an external multimodal projector.
- `full_finetuning`: train every wrapped backbone parameter.

The manifest records every trainable and frozen tensor name and count. A bare
ReCoAlign toy model has no vision encoder or LLM; real wrappers must expose those
modules under their registered names for freeze auditing.

## Per-epoch evaluation

Validation includes total loss, QA accuracy, semantic cosine, individual object /
attribute / relation / composition probe accuracy, and their aggregate structure
probe score. A training run is not evaluated from loss alone.
