# Real VLM Integration

## Scope

Phase 2.1 integrates frozen VLM backbones with the already registered EXP001–EXP003 instruments. It
does not train a checkpoint, add a learnable adapter, change a split, or introduce the ReCoAlign
method. The purpose is to run the same structured-interface interventions on real models.

## Canonical API

All implementations live under `src/recoalign/models/vlm/` and implement `BaseVLM`:

- `load()` lazily loads model resources once;
- `encode_image()` exposes visual features when the family runtime supports them;
- `prepare_input()` renders the shared prompt and records token/fact controls;
- `generate()` performs frozen deterministic decoding;
- `evaluate()` delegates to the shared answer evaluator.

Experiment code calls only `build_vlm()` and `BaseVLM`. `build_vlm()` delegates to the model
registry; it contains no per-family construction branches. Compatibility wrappers under
`models/vlm/` preserve existing imports.

## Backbones

| Registry key | Adapter | Status |
|---|---|---|
| `reference` | deterministic non-neural baseline | infrastructure only |
| `llava_1_5_7b` | LLaVA-1.5 original-checkpoint adapter | pinned, runtime dry-run ready |
| `llava_next` | LLaVA-NeXT boundary | adapter ready; checkpoint unpinned |
| `qwen_vl` | Qwen-VL/Qwen2-VL boundary | adapter ready; checkpoint unpinned |
| `internvl` | InternVL boundary | adapter ready; checkpoint/code unpinned |

Only a backbone with a pinned revision, local manifest, deterministic generation, and complete
runtime provenance can be scientific evidence. An adapter-ready row is not evidence.

The pinned LLaVA-1.5 snapshot uses the original checkpoint key layout (`model.layers.*`), not the
integrated Hugging Face LLaVA layout (`model.language_model.layers.*`). Its registered
`legacy_transformers` loader therefore composes the language model, cached CLIP vision tower, and
verified multimodal projector explicitly. Dry-run fails closed if checkpoint format and loader do
not match, if tokenizer/image processor construction fails, or if the NF4 CUDA runtime is absent.
Passing this weight-load-free check establishes runtime readiness only; it is not inference evidence.

## Resource lifecycle

The registry maintains a process-local model pool. Real model keys exclude experiment seed, so a
multi-seed run loads one frozen checkpoint once. ReferenceVLM remains seed-specific. Models are lazy,
batch size is configured, and deterministic predictions can use the content-addressed cache. LLaVA
supports processor-level batching when cache is disabled; cached execution retains per-request
lookup.

## Usage

```bash
recoalign list-vlm-models
recoalign run-vlm-eval --model llava_1_5_7b --experiment EXP001 --split test --dry-run
recoalign run-experiment --experiment EXP001 --model llava_1_5_7b --dry-run
```

Omit `--dry-run` and `--seed` for the registered multi-seed protocol. A single `--seed` is a
non-claim smoke run. The runner rejects any `--split` value different from the experiment registry.
