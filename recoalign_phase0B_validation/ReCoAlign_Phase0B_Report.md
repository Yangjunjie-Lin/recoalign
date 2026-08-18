# ReCoAlign Phase 0-B Report

Generated: 2026-08-18T01:18:48.724473+00:00

## 1. Scientific question

本实验只检验一个冻结机制问题：在 Phase 0-A 已证实 `Za` 保有语义的前提下，关系与组合语义是否在 LLaVA-1.5-7B 的 LLM hidden visual-token representations 中发生相对于 object 明显更强的线性可解码性下降。模型结构和参数均未修改；没有 adapter、训练、loss 或微调。

## 2. Protocol

- Model: official `liuhaotian/llava-v1.5-7b`, frozen Vicuna-7B LLM and official projector.
- Data: 720 balanced train + 288 balanced held-out synthetic scenes; each of 72 composition classes contributes 10/4 samples.
- Boundaries: `Za`, then mean over the same 576 visual-token positions at hidden-state indices [0, 8, 16, 24, 32]. Decision-position probes and attention are supporting diagnostics only.
- Probe: standardized multinomial logistic regression, fixed `C=1`, no nonlinear head.
- Uncertainty: 2000 paired held-out bootstrap resamples.
- Runtime precision: NF4 frozen-weight inference. This makes the run feasible on 6GB VRAM, but any positive result requires a higher-precision replication before a broad claim.
- Registered protocol validity: **FULL**.

## 3. Held-out semantic decodability

| Stage | Object | Attribute | Relation | Composition |
| --- | ---: | ---: | ---: | ---: |
| Za | 100.00% | 99.31% | 99.31% | 99.31% |
| L08_visual | 100.00% | 99.65% | 99.65% | 98.96% |
| L16_visual | 100.00% | 99.65% | 99.65% | 100.00% |
| L24_visual | 100.00% | 98.96% | 100.00% | 99.31% |
| L32_visual | 99.65% | 96.53% | 99.65% | 99.31% |

The full drops, 95% confidence intervals, and relation/composition-vs-object paired selectivity tests are recorded in `statistics.json` and `results.csv`.

### Supporting decision-position diagnostic

| Stage | Object | Attribute | Relation | Composition |
| --- | ---: | ---: | ---: | ---: |
| L00_decision | 33.33% | 16.67% | 25.00% | 1.39% |
| L08_decision | 100.00% | 95.14% | 97.57% | 97.57% |
| L16_decision | 100.00% | 97.57% | 99.65% | 100.00% |
| L24_decision | 100.00% | 100.00% | 100.00% | 100.00% |
| L32_decision | 100.00% | 99.65% | 99.65% | 100.00% |

The decision position begins at chance at state 0 because it has not yet attended to the image. By state 8 it already exposes all four semantics at 95% or better; at final state 32, Object/Attribute/Relation/Composition are 100.00%/99.65%/99.65%/100.00%. This diagnostic therefore does not reveal an accessibility failure either. It is supporting evidence, not a causal-use claim.

The layer-32 decision token assigns 15.36% mean attention mass to visual tokens. Attention is reported descriptively and is not a decision gate.

## 4. Automatic decision

Observed pattern: **no material semantic degradation inside the tested LLM layers**

Passing specific-degradation layers: `[]`.

# NO-GO

Semantic Utilization Failure is **NOT ESTABLISHED by this experiment**.

按预注册规则停止：当前证据不支持进入 Causal Intervention 设计。NO-GO 不等于证明所有自然图像或所有 VLM 都不存在 utilization failure。

## 5. Interpretation limits

- Linear decodability is a diagnostic, not proof that the model causally uses a feature.
- Attention mass is descriptive and is not treated as a causal explanation or a GO gate.
- The balanced rendered scenes isolate the hypothesized mechanism but do not establish prevalence on natural images.
- Near-ceiling accuracies bound detectable loss but may hide subtle changes smaller than the registered 15-point effect.
- NF4 and the final-four-layer fp16 CPU execution are inference approximations. A positive result would require higher-precision replication; this negative result remains scoped to the registered runtime.
