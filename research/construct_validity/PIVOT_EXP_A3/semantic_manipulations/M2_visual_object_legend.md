# M2 — Visual Object-Legend Scaffold

M2 uses four hash-ordered rows on a fixed 576×384 canvas. Every row displays a stable `E1`–`E4`
identifier, a standardized 96×96 isolated object rendering, and its shape and color text. The tile
background is RGB `(245,245,245)`.

The source generator has no registered object bounding boxes. The only permitted preflight-fixed
replacement is therefore deterministic isolated re-rendering with the frozen synthetic renderer.
No post-preflight crop or bounding-box fallback is allowed. The rendering discards original
coordinates and the row permutation is unrelated to scene spatial order.

For corruption, the complete crop-plus-shape-plus-color unit is deranged across stable entity IDs.
Dimensions, entity count, row order, and layout remain identical. Legends never display relations,
coordinates, query roles, reasoning, or answers.
