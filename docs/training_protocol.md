# ReCoAlign training protocol

Training is staged so that the interface can be audited independently of a large
backbone:

1. **Interface pretraining** on the synthetic compositional world, using graph
   annotations only as structural-consistency targets.
2. **Instruction adaptation** on a reasoning dataset while keeping the same
   visual-to-structure interface and prompt protocol.
3. **Optional task fine-tuning** for a preregistered real task.

The included `train-recoalign-toy` command runs a deterministic CPU sanity check.
It fixes one generated batch, records all three loss terms through the model API,
and writes a PyTorch checkpoint.  A tiny-batch overfit check is available with
`--overfit-check`.

Every real run should record the backbone/checkpoint hash, dataset manifest,
prompt version, seed, hardware, torch version, resolved configuration, and git
commit.  Toy results are implementation evidence only and must not be promoted to
scientific benchmark evidence.
