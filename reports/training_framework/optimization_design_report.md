# Optimization design report

The optimizer contains exactly three theory-motivated terms: semantic preservation,
structural consistency over object/attribute/relation/composition targets, and
reasoning alignment. Each stage assigns an explicit scientific role and explicit
weights; no additional data or auxiliary loss is introduced.

TRAIN001 emphasizes semantic and structural learning. TRAIN002/003 emphasize answer
alignment while retaining semantic preservation. The scheduler and freeze policy are
fully resolved in configuration and checkpointed.
