# Factorial Protocol

The main unit is a preselected scene and its unchanged image, question, options, answer evaluator,
generation settings, and tokenizer. Every unit receives all eight semantic-correctness ×
relation-correctness × serialization cells. Trial order is deterministic and randomized from the
frozen seed; inference is resumable by unique trial key, and completed or failed trials may not be
deleted. Native image-only and oracle-correct natural-language references are secondary only.

