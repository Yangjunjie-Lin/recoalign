"""Prospective primary-only power calculation with fixed Monte Carlo seeds."""

from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np

POWER_SEED = 20260825
REPLICATES = 200_000
TARGET_POWER = 0.80
ALPHA = 0.05


def calculate_primary_power() -> dict[str, Any]:
    """Recalculate all primary endpoints without reading predictions or model answers."""

    cv1 = _binomial_threshold_power(probability=0.98, threshold=0.95, seed=POWER_SEED + 1)
    gate_c = _binomial_threshold_power(probability=0.98, threshold=0.95, seed=POWER_SEED + 2)
    gate_d = _binomial_threshold_power(probability=0.95, threshold=0.90, seed=POWER_SEED + 3)
    gate_e = _separation_power(seed=POWER_SEED + 4)
    gate_f = _equivalence_power(seed=POWER_SEED)
    powers = [cv1, gate_c, gate_d, gate_e, gate_f]
    passed = min(powers) >= TARGET_POWER
    return {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A3P",
        "status": "PASS" if passed else "INCONCLUSIVE_PREINFERENCE_POWER",
        "computed_before_inference": True,
        "validation_predictions_observed": False,
        "uses_model_answers_for_planning": False,
        "secondary_generation_trials_removed": 9000,
        "primary_trial_count": 27000,
        "paired_scene_count": 500,
        "simulation": {
            "method": "paired_binomial_and_trinomial_monte_carlo",
            "replicates": REPLICATES,
            "seed": POWER_SEED,
            "alpha": ALPHA,
            "target_power": TARGET_POWER,
            "validation_seeds": 5,
            "scenes_per_seed": 100,
            "hierarchical_analysis_after_inference": (
                "fixed-seed hierarchical seed-scene percentile bootstrap"
            ),
            "multiplicity": "Holm correction across four task endpoints",
            "no_seed_deletion": True,
        },
        "endpoints": {
            "A_CV1_comprehension_per_condition": {
                "design_alternative_accuracy": 0.98,
                "required_mean": 0.95,
                "required_ci_lower": 0.90,
                "planned_n": 500,
                "estimated_power": cv1,
                "passed": cv1 >= TARGET_POWER,
            },
            "C_scaffold_comprehension_per_task": {
                "design_alternative_accuracy": 0.98,
                "required_mean": 0.95,
                "required_ci_lower": 0.90,
                "planned_n": 500,
                "estimated_power": gate_c,
                "passed": gate_c >= TARGET_POWER,
            },
            "D_scene_semantic_sufficiency_per_task": {
                "design_alternative_accuracy": 0.95,
                "required_mean": 0.90,
                "required_ci_lower": 0.85,
                "planned_n": 500,
                "estimated_power": gate_d,
                "passed": gate_d >= TARGET_POWER,
            },
            "E_intervention_separation_per_task": {
                "design_alternative": {
                    "oracle_accuracy": 0.95,
                    "corrupted_accuracy": 0.35,
                    "paired_difference": 0.60,
                    "paired_cell_probabilities": {
                        "00": 0.03,
                        "01": 0.02,
                        "10": 0.62,
                        "11": 0.33,
                    },
                },
                "required_difference": 0.50,
                "required_ci_lower_strictly_above": 0.0,
                "planned_n_pairs": 500,
                "estimated_power": gate_e,
                "passed": gate_e >= TARGET_POWER,
            },
            "F_image_interference_per_task": {
                "design_alternative_difference": 0.0,
                "equivalence_margin": [-0.05, 0.05],
                "assumed_pair_discordance": 0.10,
                "worst_case_holm_alpha": 0.0125,
                "planned_n_pairs": 500,
                "estimated_power": gate_f,
                "monte_carlo_standard_error": float(np.sqrt(gate_f * (1.0 - gate_f) / REPLICATES)),
                "passed": gate_f >= TARGET_POWER,
                "most_demanding_endpoint": True,
            },
            "A_primary_measurement_validity": {
                "planned_n": 27000,
                "required_valid_measurement_rate": 1.0,
                "stochastic_power_applicable": False,
            },
        },
        "minimum_estimated_primary_power": min(powers),
        "primary_power_passed": passed,
        "no_seed_deletion": True,
        "no_failed_trial_deletion": True,
        "no_posthoc_threshold_adjustment": True,
    }


def _binomial_threshold_power(*, probability: float, threshold: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    successes = rng.binomial(500, probability, size=REPLICATES)
    passed = successes / 500 >= threshold
    return float(np.mean(passed))


def _separation_power(*, seed: int) -> float:
    rng = np.random.default_rng(seed)
    passed = 0
    batch = 10_000
    for _ in range(REPLICATES // batch):
        cells = rng.multinomial(500, [0.03, 0.02, 0.62, 0.33], size=batch)
        difference = (cells[:, 2] - cells[:, 1]) / 500
        passed += int(np.count_nonzero(difference >= 0.50))
    return passed / REPLICATES


def _equivalence_power(*, seed: int) -> float:
    rng = np.random.default_rng(seed)
    critical = NormalDist().inv_cdf(1.0 - 0.0125)
    passed = 0
    batch = 10_000
    for _ in range(REPLICATES // batch):
        cells = rng.multinomial(500, [0.05, 0.05, 0.90], size=batch)
        difference = (cells[:, 0] - cells[:, 1]) / 500
        sum_squares = cells[:, 0] + cells[:, 1] - 500 * difference * difference
        standard_error = np.sqrt(sum_squares / 499 / 500)
        lower = difference - critical * standard_error
        upper = difference + critical * standard_error
        passed += int(np.count_nonzero((lower > -0.05) & (upper < 0.05)))
    return passed / REPLICATES


__all__ = ["calculate_primary_power"]
