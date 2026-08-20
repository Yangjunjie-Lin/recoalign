# Baseline fairness report

The baseline matrix validation passed for Original VLM, Caption Adapter, Oracle Graph
Prompt, and ReCoAlign. Controlled fields are vision backbone, LLM backbone, dataset,
split, and seed.

Oracle Graph Prompt is explicitly an upper-bound baseline. ReCoAlign records
`oracle_graph_at_inference: false`; graph supervision is training-signal-only.

This report validates comparison design, not completed real-backbone scores.
