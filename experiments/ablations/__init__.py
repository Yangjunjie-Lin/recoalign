"""Registered Phase-5 ablation experiment boundaries."""

from recoalign.analysis.mechanistic.registry import ABLATION_SPECS, validate_mechanistic_registry
from recoalign.analysis.mechanistic.runner import run_toy_mechanistic_suite

__all__ = ["ABLATION_SPECS", "run_toy_mechanistic_suite", "validate_mechanistic_registry"]
