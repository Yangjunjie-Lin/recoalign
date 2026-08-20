# ReCoAlign ablation plan

| Ablation | Implementation switch | Scientific question |
| --- | --- | --- |
| No structure token | `use_structure_tokens: false` | Is the gain due to the interface? |
| Random structure token | `random_structure_tokens: true` | Are tokens merely extra capacity? |
| Fixed graph encoder | external baseline | Is learned latent structure preferable to oracle graph encoding? |
| No structural loss | `structural_loss_weight: 0` | Does graph supervision teach the interface? |

All ablations must use identical images, prompts, seeds, optimization budgets, and
benchmark manifests.  No test-set tuning is allowed.
