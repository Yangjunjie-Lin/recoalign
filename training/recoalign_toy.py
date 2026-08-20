"""Compatibility wrapper for the source-layout toy trainer."""

from recoalign.training.recoalign_toy import (
    ToyBatch,
    checkpoint_roundtrip,
    load_recoalign_config,
    make_toy_batch,
    overfit_sanity_test,
    run_toy_training,
)

__all__ = [
    "ToyBatch",
    "checkpoint_roundtrip",
    "load_recoalign_config",
    "make_toy_batch",
    "overfit_sanity_test",
    "run_toy_training",
]
