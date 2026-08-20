# EXP001 Implementation and Reference Validation Report

Date: 2026-08-19

## Implementation report

EXP001 now executes C0 Image Only, C1 Image + Object List, C2 Image + Caption, and C3 Image + Scene
Graph on identical generated worlds. Caption and graph carry the same canonical object, attribute,
and relation facts. Every paired prediction stores the shared semantic-fact hash, semantic-unit and
relation counts, model-tokenizer counts, prompt checksum, condition, setting, seed, task, and depth.

The implementation adds natural-length and token-matched settings; accuracy by task and depths 1–4;
object/relation/compositional error taxonomy; paired bootstrap intervals and one-sided tests; graph
order, unordered-text, tuple, and JSON serialization ablations; preliminary composition-OOD
evaluation; three automatic figures; and a hashed multi-seed artifact bundle. The BaseVLM lifecycle
now exposes `prepare_input()`, `generate()`, and `evaluate()`. ReferenceVLM remains the CPU test
fixture, while the LLaVA-1.5 template explicitly opts into a local-only lazy backend.

Key locations are `experiments/graph_vs_text/runner.py`,
`experiments/graph_vs_text/analysis.py`, `models/vlm/base.py`, `models/vlm/llava.py`, and
`src/recoalign/synthetic_world/questions/conditions.py`.

## Experiment protocol

Five seeds (`20260818`–`20260822`) evaluate 120 IID scenes and a 40-scene preliminary OOD test per
seed. The primary graph-caption comparison is paired by scene. The same comparisons are repeated
after token matching, at 2-hop and 3-hop depth, and on OOD compositions. GO requires every registered
effect, confidence, p-value, positive-seed, integrity, and model-eligibility gate to pass. The full
preregistered protocol is `research/protocols/EXP001_graph_vs_text.md`.

## Reference validation result

The generated `outputs/EXP001` bundle passed all information-control checks across five seeds:

- semantic-fact hash mismatches: 0;
- incomplete caption/graph pairs: 0;
- token-matched prompt pairs per seed: 120;
- maximum caption/graph input-token delta: 1;
- artifact-manifest hash mismatches: 0.

The configured ReferenceVLM produced the following deliberately synthetic effects:

| Registered gate | Mean graph-caption effect | 95% CI | One-sided p | Gate |
|---|---:|---:|---:|:---:|
| Natural | 0.1767 | [0.1233, 0.2300] | 0.0023 | pass |
| Token matched | 0.1767 | [0.1233, 0.2300] | 0.0023 | pass |
| 2-hop token matched | 0.1800 | [0.0400, 0.3200] | 0.0438 | pass |
| 3-hop token matched | 0.1200 | [-0.0400, 0.3600] | 0.1942 | fail |
| OOD token matched | 0.1150 | [0.0600, 0.1650] | 0.0095 | pass |

The 3-hop failure is retained and reported. No best-only graph result is selected.

## Scientific interpretation

These numbers do not establish that graph gains come from structure. ReferenceVLM samples answers
from configured condition accuracies and is registered as `infrastructure_validation` with
`scientific_decision_allowed: false`; its numerical gap is therefore not an observation about a real
VLM. In addition, the preregistered 3-hop significance/CI gate failed. The only supported conclusion
is that the controlled instrument, falsification gates, provenance, and reporting path execute as
designed.

## Decision

**INCONCLUSIVE.** Do not advance to the Graph Completeness / Structural Necessity claim on this
evidence. The next scientific action is to preregister the pinned LLaVA-1.5 config as the eligible
EXP001 backend, verify local checkpoint shards, run the unchanged protocol, and accept GO or NO-GO
without modifying dataset composition or thresholds.

## Verification

- Full repository suite: 315 tests passed.
- Final EXP001/governance/synthetic targeted suite: 39 tests passed.
- Ruff: all repository checks passed.
- `git diff --check`: no whitespace errors (only platform line-ending notices).
