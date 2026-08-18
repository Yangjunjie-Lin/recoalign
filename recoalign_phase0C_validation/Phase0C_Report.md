# ReCoAlign Phase 0-C Report

## 1. Scientific question

本实验检验 Semantic Access Failure：同一行为查询的决策位置中，空间关系信息是否可由 held-out 线性 probe 解码，却未被冻结模型原有 LM head 用于同一个四分类决定。

本报告只对以下范围作结论：official LLaVA-1.5-7B、NF4 冻结推理、固定合成图像、`left/right/above/below` 下一 token 决策。它不是对所有 Vision-Language Models 的普遍性证明。

## 2. Frozen protocol

- Data: 720 balanced train + 288 held-out scenes from the immutable Phase 0-B selection.
- Availability: standardized multinomial logistic regression (`C=1.0`) on the pre-answer decision token at L00/L08/L16/L24/L32.
- Behavior: after teacher-forcing the tokenizer-identical shared answer-prefix token, argmax of the frozen LM head over the four relation tokens at their actual answer position.
- Attention: decision-to-vision attention on all 288 test scenes; supporting only.
- Intervention: decision-token activation blending at L08/L16/L24. The primary causal test was fixed at L16, α=1.00, before results.
- Uncertainty: 2000 fixed-seed bootstrap resamples.
- Forbidden operations: no fine-tuning, model-structure change, adapter, or added loss.

## 3. Availability–utilization result

| State | Availability | Behavior | Gap | Availability on behavior errors |
| --- | ---: | ---: | ---: | ---: |
| L00 | 25.00% | 38.19% | -13.19% | 0.56% |
| L08 | 99.65% | 38.19% | 61.46% | 99.44% |
| L16 | 100.00% | 38.19% | 61.81% | 100.00% |
| L24 | 100.00% | 38.19% | 61.81% | 100.00% |
| L32 | 100.00% | 38.19% | 61.81% | 100.00% |

Primary L32 held-out availability was 100.00% (95% CI 100.00%–100.00%); behavioral utilization was 38.19%. The paired gap was 61.81% (95% CI 56.25%–67.36%). There were 178 behavioral errors; the L32 probe was correct on 100.00% of them.

## 4. Attention and causal intervention

Attention is descriptive and cannot establish causal use. Per-image, per-layer values and correct/incorrect contrasts are in `results/attention_analysis.csv` and `results/statistics.json`.

The primary L16 full positive-donor patch recovered 8.33% of eligible baseline failures. The counterfactual control recovered 0.00%; the paired positive-minus-control advantage was 8.33% (95% CI 0.00%–20.83%).

The registered secondary L24 full patch produced a larger positive-minus-control effect of 45.83% (95% CI 25.00%–66.67%). This is a strong hypothesis-generating late-layer signal, but L24 was not the run-before-results primary causal gate and therefore cannot replace the failed L16 test post hoc.

## 5. Registered decision gates

- [x] `full_registered_protocol`
- [x] `representation_available`
- [x] `availability_behavior_gap`
- [x] `representation_available_on_failures`
- [ ] `causal_patch_recovery`

# NO-GO

Semantic Access Failure is **NOT ESTABLISHED** in the registered experimental scope.

ReCoAlign 机制发现路线终止。

## 6. Interpretation limits

- A linear probe demonstrates decodability, not by itself causal use; this is why the registered causal-patch gate is mandatory.
- The matched donor patch changes a complete decision-token activation, not an isolated neuron or uniquely identified circuit.
- NF4 is an inference approximation. Any GO requires bf16/fp16 replication before a broad mechanism claim.
- Synthetic shapes isolate the relation variable but do not estimate prevalence on natural-image VLM tasks.
- The supplied task description omitted Sections 3–12. Protocol v1 was frozen before any Phase 0-C output; v2/v3 are measurement-integrity corrections documented below. The final protocol is recorded by hash in `decision.json`.

## 7. Assay-integrity correction

An initial execution compared the four relation-token logits before Vicuna's mandatory shared answer-space token. It produced a fixed `above` output (25% accuracy) and no eligible same-class successful donor. Tokenizer inspection showed that every actual candidate is encoded as the identical space token `29871` followed by its relation token. That execution was invalidated before scientific adjudication. Protocol v2 corrected only the answer position. Its audit then found one near-tie receiver whose unpatched answer flipped because behavior-with-attention used eager attention while patching used SDPA. Protocol v3 puts attention in a separate supporting forward, keeps behavior/hidden/patch on the same SDPA path, and adds exact baseline reproduction as an integrity gate. Data, thresholds, layers, bootstrap count, donor rule, and substantive GO gates are unchanged.
