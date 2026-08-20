# ReCoAlign method design

## Motivation

EXP004 motivates a learnable interface between visual semantics and language
reasoning.  The current ReferenceVLM result is an infrastructure sanity check,
not evidence about a real VLM; real-model efficacy remains an open evaluation.

ReCoAlign therefore learns an intermediate representation from visual tokens.  It
does not append an oracle scene graph to the prompt.

## Architecture

```text
image / visual tokens
        |
visual projection + typed latent queries
        |
cross-attention structure tokens
        |
gated projection to LLM hidden space
        |
reasoning context and answer head
```

Structure tokens are latent slots.  The type layout (`object`, `attribute`,
`relation`, `composition`) is only a weak inductive bias; slots are not graph
nodes and do not require a hand-authored ordering.

## Objectives

The method has exactly three objectives:

1. **Semantic preservation loss** keeps structure summaries aligned with pooled
   visual features (and optional semantic targets).
2. **Structural consistency loss** uses object/relation/attribute annotations as
   training supervision for a structural head.
3. **Reasoning alignment loss** trains the adapted context against answer labels.

Graph annotations are training signals only.  `ReCoAlignModel.forward` accepts
visual tokens and has no graph input, which is the no-oracle-input contract.

## Baselines and ablations

The evaluation plan retains the original VLM, oracle graph prompting, and caption
reasoning baselines.  Ablations disable structure tokens, replace them with random
tokens, use a fixed graph encoder, or set the structural loss weight to zero.

## Scientific status

Toy training and checkpoint tests validate implementation mechanics.  They do not
establish gains on EXP001–EXP003 or on real VLMs.  Those claims require a complete
training/evaluation run with frozen benchmark manifests and pre-registered seeds.
