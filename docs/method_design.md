# ReCoAlign method design

## Status: inactive historical design

The frozen EXP001/EXP004 evidence rejected the diagnosis that motivated this design. This file is
retained to document existing code and toy tests; it is not the active research direction and does
not authorize real-model training. Any future intervention must be derived from a replicated pivot
diagnosis and receive a new method freeze.

## Motivation

The original design assumed that EXP004 would reveal high semantic availability and low structured
access. The real LLaVA diagnosis instead reported low SAS and a NO-GO decision, so that motivation
is falsified for the frozen scope.

The implementation below explores an intermediate representation from visual tokens and does not
append an oracle scene graph to the prompt. It remains an engineering artifact only.

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
