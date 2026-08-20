# VLM Integration Report

## Outcome

The Phase-2.1 execution platform is implemented without changing EXP001–EXP003. A canonical
`BaseVLM` lifecycle, shared prompt, answer evaluator, model registry, lazy model pool, deterministic
cache, batch boundary, provenance capture, CLI runner, and failure analysis now connect all three
experiments.

## Supported models

| Model | Implementation status | Evidence status |
|---|---|---|
| ReferenceVLM | executable | non-neural infrastructure baseline |
| LLaVA-1.5-7B | pinned original checkpoint; legacy loader; runtime dry-run ready | eligible after complete inference |
| LLaVA-NeXT | common adapter boundary | checkpoint/revision not pinned |
| Qwen-VL | common adapter boundary | checkpoint/revision not pinned |
| InternVL | common adapter boundary | checkpoint/code revision not pinned |

All rows implement the same interface and declare EXP001, EXP002, and EXP003 compatibility. Only
LLaVA-1.5 is currently registered as potentially claim-bearing real-model evidence.

## Validation performed

- Mock LLaVA backend: lazy load exactly once, generation and image encoding passed.
- ReferenceVLM: EXP001 single-seed runner completed 1,600 predictions and failure analysis.
- LLaVA-1.5 weight-load-free dry-run: 9/9 file hashes, legacy checkpoint/loader compatibility,
  Llama tokenizer, CLIP image processor, required dependencies, and CUDA/NF4 hardware passed;
  `weights_loaded=false` and zero predictions were recorded.
- Prompt: the committed shared protocol is byte-equivalent to Phase 1.
- Configuration: JSON Schema plus semantic deterministic-generation checks pass/fail closed.
- Governance: Reference remains infrastructure-only; pinned LLaVA is a permitted scientific backend.
- Regression: full repository suite passed 341/341; the final provenance/trackability change passed
  23 focused VLM and structured-interface tests; Ruff and `git diff --check` passed.
- Real-weight smoke: LLaVA-1.5 loaded the frozen 7B NF4 checkpoint, CLIP tower, and projector on the
  RTX 3060 CUDA environment; image-only, caption, and graph prompts each generated a normalized
  correct answer for one controlled sample.
- Real EXP001 attempt: 629/1,600 deterministic requests were cached before one CUDA generation made
  no progress for more than six minutes. The run was safely interrupted; no partial predictions or
  metrics were promoted, and the incomplete dataset/config was archived under `.local_artifacts`.

No real-model accuracy claim is made by this report. The single-sample smoke proves loader and
inference-path viability only; the interrupted EXP001 attempt is not a statistical result.
`runtime_ready` means that frozen inference can start under the pinned environment, not that a
complete experiment has run.
