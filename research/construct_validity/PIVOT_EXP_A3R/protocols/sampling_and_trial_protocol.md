# PIVOT_EXP_A3R Sampling and Trial Protocol

The five validation seeds, 500 selected scenes, images, choice order, correct-choice mapping, and
all 27,000 primary trials are inherited byte-for-byte at the protected-field level from the frozen
PIVOT_EXP_A3 v1 inventory. No scene generation or selection is repeated. The 9,000 secondary
prompts are re-rendered before inference solely to end at the frozen ASCII suffix
`FINAL_CHOICE=`. Their keys append
`response_contract=answer-contract-v3-continuation`; primary trial keys remain unchanged.

For each of five new validation seeds, the frozen generator produces 1,200 candidates without
loading the VLM. Records with exactly four objects are sorted by scene ID and the first 100 are
selected. Development uses eight four-object scenes from separate seed `20260830`. Development and
validation IDs are hashed and checked against all A2 source scene IDs.

Every selected scene produces four semantic tasks and the preregistered factorial cells. Choices
come only from the four scene objects and their ontology values. Correct scene-truth positions are
balanced 25/25/25/25 within seed × task × manipulation. Trial keys include study, seed, scene,
construct, manipulation, evidence truth, image context, task, and response method and must be
unique. A3R verifies primary prompt SHA, image path and SHA, choices, correct choice, and trial key
against every corresponding A3 v1 row before it can freeze.
