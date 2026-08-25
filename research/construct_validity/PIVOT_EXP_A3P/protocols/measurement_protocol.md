# Primary Measurement Protocol

Every frozen prompt ends in `FINAL_CHOICE=`. LLaVA-1.5-7B is evaluated in NF4 inference mode for
the no-leading-space single-token continuations 1, 2, 3, and 4. Exactly four finite conditional log
likelihoods are required. The registered-ID argmax is selected; exact ties retain numeric order.

The scorer accepts prompt, image, and the four IDs only. Truth and correctness are joined after the
score result has been persisted. No response-generation or response-interpretation surface exists in
the primary-only runner, and no model parameter is updated.

The registered runtime strategy is
`staged_gpu_vision_feature_cache_then_nf4_language_forward_v1`. Vision and projector parameters
execute on CUDA and emit exact projected image-token tensors keyed by image/model SHA. They are
released before the CUDA NF4 language forward so the two parameter groups do not compete for the
6 GB device. No model parameter is executed on CPU. Parent development score anchors across M0,
M1, and M2 must reproduce exactly; otherwise the scorer is treated as drifted and freeze fails.
