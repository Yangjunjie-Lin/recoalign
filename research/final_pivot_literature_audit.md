# Final Pivot Literature Novelty Audit

Audit cutoff: **2026-08-25**. Sources were restricted to arXiv originals, formal venue records, and
official project/paper pages. The closest papers were read beyond titles: abstracts plus available
HTML sections covering problem definition, interventions, metrics, experiments, and conclusions.

```yaml
novelty_status: VERIFIED_NO_CANDIDATE_PASSES
authorized_candidate: null
```

The audit tests novelty; it does not upgrade ReCoAlign's frozen evidence.

## PH005 — Arbitrary Referential Binding Limitation

| Primary work | Problem definition | Claimed mechanism | Experimental intervention | Model family | Dataset setting | Main conclusion | Difference from PH005 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Symbol tuning improves in-context learning in language models, EMNLP 2023](https://arxiv.org/abs/2305.08298) | Learn mappings when natural-language labels are replaced with arbitrary symbols. | Models must use in-context input–label mappings when semantic label shortcuts are removed. | Replace labels with random words/characters/integers; test unseen tasks and flipped labels; symbol-tune four models. | Flan-PaLM 8B–540B | 22 NLP tuning datasets, 11 unseen ICL tasks, algorithmic suites | Symbol tuning improves reasoning over arbitrary symbols and following flipped labels, though some results remain near chance. | PH005 applies arbitrary IDs to visual entities rather than task labels, but the core arbitrary-symbol mapping limitation and scaling logic are already explicit. |
| [Diagnosing Dense Same-Class Attribute Misbinding in LVLMs](https://arxiv.org/abs/2608.16805) | Distinguish attribute recognition from assigning a visible attribute to the wrong instance. | Local same-class competition produces source-identifiable instance–attribute transfers. | Four binding question levels; source-instance error attribution; crop/context and instance-first interventions. | Five open-source plus two API LVLMs, including LLaVA-1.5 | 524 real images, 1,773 instances, 9,580 questions | Misbinding is measurable across seven LVLMs; transfers concentrate on adjacent instances; interventions are model-dependent. | PH005 uses arbitrary textual aliases instead of spatial same-class instances, but the broader “recognize values yet bind them to the wrong entity” problem, metrics, LLaVA result, and interventions are already directly covered. |
| [Evaluating Compositional Generalisation in VLMs and Diffusion Models](https://arxiv.org/abs/2508.20783) | Bind objects to attributes and relations under zero-shot and generalized zero-shot composition. | Bag-of-words-like representations and similar relational embeddings impede binding. | Controlled object–attribute and relation binding tasks under ZSL/GZSL. | CLIP, ViLT, diffusion classifier | Compositional binding benchmarks | All tested model classes struggle on relational GZSL; concept binding is an established evaluation target. | PH005 adds arbitrary referential names, but its claimed scientific center remains adjacent to established binding diagnostics. |

Adjudication: novelty is **insufficient**. PH005 is not merely a clean independent consequence of
frozen evidence: it was named after M1/A3R failures, while A3P does not separate alias mapping from
general evidence ingestion. A minimal direct > single-alias > multi-alias test is feasible, but
direct support is 2/5, post-hoc risk is 5/5, and the closest 2026 VLM paper already provides a
multi-model binding benchmark with source-aware metrics and interventions.

## PH006 — Serialization-Conditioned Evidence Usability

| Primary work | Problem definition | Claimed mechanism | Experimental intervention | Model family | Dataset setting | Main conclusion | Difference from PH006 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Promptception: How Sensitive Are Large Multimodal Models to Prompts?, EMNLP 2025](https://arxiv.org/abs/2509.03986) | Quantify LMM MCQA sensitivity to prompt phrasing and structure. | Instruction following and prompt formulation alter response behavior; sensitivity differs between open and proprietary models. | 61 prompt types in 15 categories, including choice formatting, structured formatting, position, length, and answer handling. | Eight open-source and two proprietary LMMs | MMStar, MMMU-Pro, MVBench | Minor phrasing/structure changes can shift accuracy by up to 15%; prompting affects benchmark stability. | PH006 restricts the manipulation to semantically equivalent evidence serializations, but it has not isolated a mechanism beyond this established prompt/format sensitivity. |
| [Modeling Variants of Prompts for Vision-Language Models](https://arxiv.org/abs/2503.08229) | Make VLM performance robust across prompt-template variants. | Prompt template design is a major source of performance variance. | Hundreds of templates across six types; a VAE models prompt-structure variants for robustness. | CLIP-style VLMs | 11 downstream datasets | Prompt-structure robustness can be benchmarked and improved, confirming format dependence is not new. | PH006 concerns generative VLM evidence consumption rather than CLIP classification, but “serialization usability” needs a sharper mechanism to exceed prompt-template robustness. |

Frozen ReCoAlign observations—Caption > Graph, JSON > natural language > triples, and M1 > M2—do
not form a controlled common law. They differ in information channel, construct, visual content,
tokenization, and task. PH006 therefore cannot currently distinguish pretrained serialization
familiarity from generic prompt sensitivity. Its direct support is 3/5, identifiability 3/5,
post-hoc risk 4/5, and researcher-degrees-of-freedom risk 4/5. Novelty is **insufficient**.

## PH007 — Cross-Modal Evidence Competition

| Primary work | Problem definition | Claimed mechanism | Experimental intervention | Model family | Dataset setting | Main conclusion | Difference from PH007 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Which Source Wins? Task-Dependent Reliance in Vision-Language Models](https://arxiv.org/abs/2608.17205) | Measure how VLMs reallocate reliance when visual and textual evidence conflict and one source degrades. | Modality reliance is task-, evidence-, model-, evaluation-, and prompt-dependent rather than a fixed preference. | Counterbalanced conflicting sources; four image/text legibility levels; generated answers and length-normalized CLL margins; chart-to-table control. | Six open-weight CLL models plus behavioral InternVL and two frontier APIs | GSM8K, SVAMP, ChartQA-Conflict | Reallocation direction reverses between arithmetic and chart settings; no universal modality winner exists. | PH007 asks whether vision interferes with supplied semantic evidence, but this work already provides the controlled degradation and likelihood design needed to identify source competition across models. |
| [SIGNPOST-Bench: Benchmarking Text-Vision Conflict Resolution in MLLMs](https://arxiv.org/abs/2608.04244) | Evaluate arbitration between visual scenes and conflicting embedded text. | Compatible, unrelated, and adversarial text shifts grounded visual decisions in distinct ways. | Original/blank/similar/random/adversarial counterfactual quintuplets with localized scene-text interventions. | 20 MLLMs from seven providers | 5,111 groups and 25,555 geolocation image variants | Every model shifts toward injected conflicting text; robustness is not predicted by clean accuracy. | PH007 supplies external semantic evidence rather than editing scene text, but multimodal conflict and competition are already tested at much broader cross-model scale. |
| [Sycophancy in Vision-Language Models](https://arxiv.org/abs/2408.11261) | Analyze models following misleading textual suggestions over visual evidence. | Language-side suggestion can override or bias grounded visual judgment. | Systematic misleading-text conditions and inference-time mitigation. | Vision-language models | Multiple visual tasks; journal version in Neurocomputing 2026 | Cross-modal textual influence on visual decisions is an established failure mode. | PH007's clean-evidence interference is narrower, but the general modality-competition claim is not novel without a separately identified mechanism. |

M1 Gate F failed, whereas M2 Gate F passed only while M2 stayed near chance and failed Gates C, D,
and E. This cannot exclude floor effects or unusable evidence. Direct support is 2/5, novelty 1/5,
and post-hoc risk 4/5. The adjacent literature is broader, multi-model, and more causally controlled.
Novelty is **insufficient**.

## Novelty-gate decision

No candidate has both an independently supported ReCoAlign mechanism and a sufficiently distinct
position against the primary literature. PH005 and PH006 are adjacent to mature binding/symbol and
prompt-robustness work; PH007 substantially overlaps current modality-arbitration benchmarks.

```yaml
PH005: FAIL
PH006: FAIL
PH007: FAIL
candidate_authorized_for_execution: null
```

