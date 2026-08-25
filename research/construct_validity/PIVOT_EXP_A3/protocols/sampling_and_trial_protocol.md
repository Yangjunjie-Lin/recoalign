# Sampling and Trial Protocol

For each of five new validation seeds, the frozen generator produces 1,200 candidates without
loading the VLM. Records with exactly four objects are sorted by scene ID and the first 100 are
selected. Development uses eight four-object scenes from separate seed `20260830`. Development and
validation IDs are hashed and checked against all A2 source scene IDs.

Every selected scene produces four semantic tasks and the preregistered factorial cells. Choices
come only from the four scene objects and their ontology values. Correct scene-truth positions are
balanced 25/25/25/25 within seed × task × manipulation. Trial keys include study, seed, scene,
construct, manipulation, evidence truth, image context, task, and response method and must be
unique.
