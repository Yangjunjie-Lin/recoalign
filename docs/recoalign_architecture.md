# ReCoAlign architecture reference

The canonical implementation is under `src/recoalign/models/`:

- `interface/structure_tokens.py`: learnable latent query bank.
- `interface/structure_encoder.py`: visual projection and cross-attention.
- `interface/adapter.py`: gated visual + structure context in LLM space.
- `recoalign/model.py`: end-to-end module, heads, losses, and checkpoint contract.

The method is intentionally small.  It adds no new vision backbone, LLM, adapter
training checkpoint, or oracle graph input path.  Existing VLM backbones remain
usable as frozen feature providers.

`extract_structure_tokens(visual_tokens)` is the diagnostic boundary.  Its output
shape is `[batch, num_structure_tokens, interface_dim]`, and its output can be
visualized or consumed by a downstream language model.
