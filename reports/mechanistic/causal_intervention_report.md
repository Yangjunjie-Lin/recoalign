# Causal intervention report

Three inference interventions are implemented:

- remove: zero all structure slots;
- shuffle: independently permute slots within each sample;
- replace: exchange slots across samples.

Every intervention preserves dimensions, does not mutate the baseline tensor, and
records accuracy delta, prediction changes, and token displacement. Real-VLM causal
claims remain pending because the registered backbones do not yet expose learned
hidden-context injection.
