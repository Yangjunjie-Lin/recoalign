# Ablation execution plan

The four ablation configs are executable through `run-recoalign-ablation`. The
controlled sanity suite also executes clean-label, randomized-label, no-structure,
and checkpoint-resume conditions on the same generated visual-token split.

Execution order for a real backbone:

1. Freeze dataset and checkpoint manifests.
2. Run the unmodified ReCoAlign configuration for all preregistered seeds.
3. Run one-factor ablations with the same optimization budget.
4. Evaluate with the unchanged EXP001–EXP003 and external benchmark protocols.
5. Publish every split, including failed or neutral outcomes.
