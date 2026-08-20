# Dataset Generation Report

## Validated artifact

The reference example dataset was generated with `configs/synthetic_benchmark.yaml` on 2026-08-19.

| Field | Value |
| --- | --- |
| generator | `recoalign.synthetic_world.v2` |
| renderer | `recoalign.synthetic_renderer.v2` |
| seed | 20260819 |
| samples | 1,000 |
| resolution / style | 192 × 192 / flat |
| split strategy | composition |
| materialized files | 5,005 |
| artifact size | 21,383,201 bytes (20.39 MiB) |
| dataset JSONL SHA-256 | `d728c90317c66e0f612ceaf2db846f661674b6c06651175e93e048cde0f67e79` |
| validation | PASS |

The reproducible artifact is at `outputs/synthetic_world_example/` and is intentionally ignored by
Git. Its tracked config, schema, manifest declaration, code, and this aggregate report are sufficient
to regenerate it.

## Question and complexity distribution

| Question family | Count |
| --- | ---: |
| object reasoning | 250 |
| relation reasoning | 250 |
| attribute reasoning | 250 |
| multi-hop reasoning | 250 |

| Required hop depth | Count |
| --- | ---: |
| 1 | 750 |
| 2 | 84 |
| 3 | 83 |
| 4 | 83 |

One-hop samples cover object, relation, and attribute queries. Multi-hop samples are balanced across
2–4 licensed transitive paths.

## Relation distribution

| Relation | Count | Relation | Count |
| --- | ---: | --- | ---: |
| left | 104 | right | 105 |
| above | 105 | below | 105 |
| front | 104 | behind | 104 |
| near | 63 | far | 62 |
| inside | 62 | contains | 62 |
| touching | 62 | holding | 62 |

Directional/depth relations have additional mass because they alone support valid multi-hop
questions. All twelve ontology relations occur in direct reasoning samples.

## Composition split audit

| Partition | Count |
| --- | ---: |
| train | 456 |
| validation | 81 |
| test | 463 |

The validator found zero ordered composition signatures shared between train and test. It also
recomputed every answer, regenerated every caption and information-equivalence hash, validated all
1,000 schemas, verified every image checksum, and confirmed unique sample IDs.

## Reproduction command

```powershell
recoalign generate-synthetic `
  --config configs/synthetic_benchmark.yaml `
  --count 1000 `
  --output outputs/synthetic_world_example
```
