# Real VLM Execution Status

## LLaVA-1.5-7B

The pinned original-format checkpoint was loaded with the registered `legacy_transformers` loader
on the local CUDA environment. The CLIP vision tower and verified multimodal projector loaded, and a
single controlled sample produced valid normalized answers for image-only, caption, and graph inputs.

The first registered EXP001 seed (`20260818`) then started with the unchanged five-seed config,
shared prompt, deterministic generation, and all pre-registered conditions. The content-addressed
cache recorded 629 of 1,600 expected requests. One subsequent CUDA generation made no cache progress
for more than six minutes while occupying the full 6GB device, so the process was safely interrupted.

This is retained as an incomplete execution attempt, not a benchmark result:

- no `predictions.jsonl` was promoted;
- no `metrics.json` or decision report was generated;
- no partial accuracy is used for scientific claims;
- the generated dataset/config are archived under `.local_artifacts/vlm_validation/`;
- cached responses remain available for a future larger-memory retry.

The real-model evidence gate therefore remains `INCONCLUSIVE`. The implementation gate is `GO`.
