# M1 — Canonical Entity Table

The exact syntax is one header followed by four rows:

```text
ENTITY_TABLE
E1 | shape=<registered shape> | color=<registered color>
E2 | shape=<registered shape> | color=<registered color>
E3 | shape=<registered shape> | color=<registered color>
E4 | shape=<registered shape> | color=<registered color>
```

Row order is a scene-bound hash permutation and has no spatial or query-role meaning. Oracle and
corrupted tables use identical entity IDs, row order, fact count, and punctuation. Corruption moves
complete shape-color pairs by a fixed derangement and is rejected if any binding remains true.
