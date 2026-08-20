# Comprehensive multi-VLM evaluation

## Scientific purpose

Phase 4 evaluates whether ReCoAlign improves compositional reasoning across models
and tasks while preserving general capabilities. It does not treat benchmark score
aggregation as mechanism evidence. Every claim must connect frozen EXP001–EXP004
results, a trained interface checkpoint, and paired benchmark outcomes.

## Unified boundary

All generative benchmark cells use `BaseVLM`. Dataset adapters normalize SugarCrepe,
ARO, Winoground, CREPE, GQA, and MMVP into image/question/choice/answer records.
Model-specific prompt or metric branches are prohibited. The same prompt,
temperature, decoding, split, evaluator, and seed set are resolved in configuration.

Existing OpenCLIP retrieval and official compositional metrics remain frozen. They
are capability-preservation evidence and are not silently replaced by generative QA
metrics.

## ReCoAlign execution gate

`generate_with_interface` is a hidden-context runtime boundary. It cannot serialize
structure tokens into text. A ReCoAlign matrix cell is blocked unless:

1. the trained interface checkpoint exists and is hashed;
2. the backbone runtime implements learned-token context injection;
3. the benchmark manifest and frozen annotations exist;
4. the registered deterministic decoding protocol validates.

LLaVA-NeXT, Qwen-VL, and InternVL currently have adapter boundaries but no pinned
executable checkpoint in this workspace. Their cells remain visible as blocked.

## Matrix and reporting

The minimum matrix contains four model families, EXP001–EXP003, six external
benchmarks, four main methods, and three seeds: 432 registered cells. Five seeds are
reserved for final paper evidence. Missing, blocked, failed, and negative cells are
retained in reports.

Use:

```text
recoalign validate-benchmark-matrix
recoalign run-vlm-benchmark --config CONFIG --dry-run
recoalign build-comprehensive-report
```
