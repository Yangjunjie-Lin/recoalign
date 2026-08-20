"""Compatibility import for the authoritative src/recoalign implementation."""

from recoalign.synthetic_world.generator.generator import (
    GeneratorConfig,
    SyntheticWorldGenerator,
    write_jsonl,
)

__all__ = ["GeneratorConfig", "SyntheticWorldGenerator", "write_jsonl"]
