"""Compatibility runner for the mechanistic ablation suite."""

from recoalign.analysis.mechanistic.runner import run_toy_mechanistic_suite


def run(*, output_dir: str = "reports/mechanistic", seeds: tuple[int, ...] = (101, 202, 303)):
    return run_toy_mechanistic_suite(output_dir=output_dir, seeds=seeds)


__all__ = ["run"]
