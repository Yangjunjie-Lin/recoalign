# EXP004 Mechanism Diagnosis Report

The EXP004 diagnostic framework is implemented and executed end-to-end with the registered
ReferenceVLM infrastructure baseline. It measures visual semantic availability, structured
representation accessibility, and reasoning execution without training, adapter changes, prompt
tuning, or benchmark changes.

The ReferenceVLM result is intentionally not scientific VLM evidence. Its deterministic image
representation is a path-derived sanity vector, so the observed low SAS is a validation of the
diagnostic guardrail, not evidence that a real VLM lacks visual semantics. The real-model diagnosis
remains pending a complete eligible-backbone run.

The authoritative machine-readable artifacts are under `outputs/EXP004/`.
