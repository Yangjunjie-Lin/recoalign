"""Dependency-light, deterministic statistics for falsifiable experiment decisions."""

from __future__ import annotations

import itertools
import math
import random
import statistics
from collections.abc import Sequence
from typing import Any

DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_SEEDS = 3
KEY_EXPERIMENT_SEEDS = 5


def bootstrap_confidence_interval(
    values: Sequence[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    samples: int = 10_000,
    seed: int = 7,
) -> tuple[float, float]:
    """Return a percentile bootstrap interval for the arithmetic mean."""

    numeric = _finite_values(values)
    if not numeric:
        raise ValueError("bootstrap confidence interval requires at least one finite value")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if len(numeric) == 1:
        return numeric[0], numeric[0]

    rng = random.Random(seed)
    size = len(numeric)
    estimates = sorted(
        sum(numeric[rng.randrange(size)] for _ in range(size)) / size for _ in range(samples)
    )
    alpha = 1.0 - confidence
    return (
        _quantile(estimates, alpha / 2.0),
        _quantile(estimates, 1.0 - alpha / 2.0),
    )


def paired_t_test(
    first: Sequence[float],
    second: Sequence[float],
    *,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Run an exact Student paired t-test without a SciPy runtime dependency."""

    differences = _paired_differences(first, second)
    if len(differences) < 2:
        raise ValueError("paired t-test requires at least two pairs")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be two-sided, greater, or less")

    count = len(differences)
    mean = sum(differences) / count
    variance = sum((value - mean) ** 2 for value in differences) / (count - 1)
    if variance == 0.0:
        statistic = 0.0 if mean == 0.0 else math.copysign(math.inf, mean)
    else:
        statistic = mean / math.sqrt(variance / count)
    cdf = _student_t_cdf(statistic, count - 1)
    if alternative == "greater":
        p_value = 1.0 - cdf
    elif alternative == "less":
        p_value = cdf
    else:
        p_value = 2.0 * min(cdf, 1.0 - cdf)
    return {
        "test": "paired_t_test",
        "alternative": alternative,
        "statistic": statistic if math.isfinite(statistic) else None,
        "p_value": min(1.0, max(0.0, p_value)),
        "degrees_of_freedom": count - 1,
        "n": count,
        "mean_difference": mean,
        "zero_variance": variance == 0.0,
    }


def permutation_test(
    first: Sequence[float],
    second: Sequence[float],
    *,
    alternative: str = "two-sided",
    samples: int = 10_000,
    seed: int = 7,
    exact_threshold: int = 16,
) -> dict[str, Any]:
    """Run a paired sign-flip permutation test, exactly for small samples."""

    differences = _paired_differences(first, second)
    if not differences:
        raise ValueError("permutation test requires at least one pair")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be two-sided, greater, or less")
    if samples <= 0:
        raise ValueError("samples must be positive")

    observed = sum(differences) / len(differences)
    exact = len(differences) <= exact_threshold
    if exact:
        assignments = itertools.product((-1.0, 1.0), repeat=len(differences))
        estimates = [
            sum(sign * value for sign, value in zip(signs, differences, strict=True))
            / len(differences)
            for signs in assignments
        ]
    else:
        rng = random.Random(seed)
        estimates = [
            sum((1.0 if rng.getrandbits(1) else -1.0) * value for value in differences)
            / len(differences)
            for _ in range(samples)
        ]

    extreme = sum(_is_extreme(value, observed, alternative) for value in estimates)
    denominator = len(estimates)
    p_value = extreme / denominator if exact else (extreme + 1) / (denominator + 1)
    return {
        "test": "paired_permutation_test",
        "alternative": alternative,
        "statistic": observed,
        "p_value": p_value,
        "n": len(differences),
        "permutations": denominator,
        "exact": exact,
    }


def paired_bootstrap_test(
    first: Sequence[float],
    second: Sequence[float],
    *,
    alternative: str = "greater",
    samples: int = 10_000,
    seed: int = 7,
) -> dict[str, Any]:
    """Test a paired mean difference with a centered non-parametric bootstrap.

    Pairing is preserved on every draw.  The null distribution is formed by centering the
    observed paired differences at zero; the reported interval is computed from the uncentered
    paired differences.  The plus-one correction prevents a zero Monte-Carlo p-value.
    """

    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be greater, less, or two-sided")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    differences = _paired_differences(first, second)
    if not differences:
        raise ValueError("paired bootstrap requires at least one pair")
    observed = statistics.fmean(differences)
    centered = [value - observed for value in differences]
    rng = random.Random(seed)
    null_statistics: list[float] = []
    interval_statistics: list[float] = []
    size = len(differences)
    for _ in range(samples):
        indices = [rng.randrange(size) for _ in range(size)]
        null_statistics.append(statistics.fmean(centered[index] for index in indices))
        interval_statistics.append(statistics.fmean(differences[index] for index in indices))
    extreme = sum(_is_extreme(value, observed, alternative) for value in null_statistics)
    low, high = _interval_from_sorted(interval_statistics, DEFAULT_CONFIDENCE_LEVEL)
    return {
        "method": "paired_bootstrap",
        "statistic": observed,
        "p_value": (extreme + 1) / (samples + 1),
        "alternative": alternative,
        "samples": samples,
        "seed": seed,
        "n_pairs": size,
        "confidence_interval": {
            "level": DEFAULT_CONFIDENCE_LEVEL,
            "lower": low,
            "upper": high,
        },
    }


def multiple_seed_summary(
    values: Sequence[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    bootstrap_samples: int = 10_000,
    seed: int = 7,
    significance_test: str = "paired_t_test",
) -> dict[str, Any]:
    """Summarize a scalar effect across independent random seeds."""

    numeric = _finite_values(values)
    if not numeric:
        raise ValueError("multiple-seed summary requires at least one finite value")
    low, high = bootstrap_confidence_interval(
        numeric,
        confidence=confidence,
        samples=bootstrap_samples,
        seed=seed,
    )
    if significance_test == "paired_bootstrap":
        test = paired_bootstrap_test(
            numeric,
            [0.0] * len(numeric),
            alternative="greater",
            samples=bootstrap_samples,
            seed=seed + 1,
        )
    elif significance_test == "paired_t_test":
        test = (
            paired_t_test(numeric, [0.0] * len(numeric), alternative="greater")
            if len(numeric) >= 2
            else None
        )
    else:
        raise ValueError(f"unsupported significance test: {significance_test}")
    return {
        "mean": sum(numeric) / len(numeric),
        "std": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
        "confidence_interval": {
            "level": confidence,
            "method": "percentile_bootstrap",
            "lower": low,
            "upper": high,
            "samples": bootstrap_samples,
        },
        "statistical_test": test,
        "n_seeds": len(numeric),
        "positive_seed_fraction": sum(value > 0.0 for value in numeric) / len(numeric),
        "values": numeric,
    }


def _paired_differences(first: Sequence[float], second: Sequence[float]) -> list[float]:
    if len(first) != len(second):
        raise ValueError("paired samples must have equal length")
    left = _finite_values(first)
    right = _finite_values(second)
    if len(left) != len(first) or len(right) != len(second):
        raise ValueError("paired samples must contain only finite numeric values")
    return [a - b for a, b in zip(left, right, strict=True)]


def _finite_values(values: Sequence[float]) -> list[float]:
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("statistical inputs must be numeric and not boolean")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("statistical inputs must be finite")
        result.append(numeric)
    return result


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _interval_from_sorted(values: Sequence[float], confidence: float) -> tuple[float, float]:
    ordered = sorted(values)
    alpha = (1.0 - confidence) / 2.0
    return _quantile(ordered, alpha), _quantile(ordered, 1.0 - alpha)


def _is_extreme(value: float, observed: float, alternative: str) -> bool:
    tolerance = 1e-15
    if alternative == "greater":
        return value >= observed - tolerance
    if alternative == "less":
        return value <= observed + tolerance
    return abs(value) >= abs(observed) - tolerance


def _student_t_cdf(statistic: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    if statistic == math.inf:
        return 1.0
    if statistic == -math.inf:
        return 0.0
    if statistic == 0.0:
        return 0.5
    x = degrees_of_freedom / (degrees_of_freedom + statistic * statistic)
    tail = 0.5 * _regularized_incomplete_beta(x, degrees_of_freedom / 2.0, 0.5)
    return 1.0 - tail if statistic > 0.0 else tail


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must be in [0, 1]")
    if x in {0.0, 1.0}:
        return x
    log_term = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_term)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(x, a, b) / a
    return 1.0 - front * _beta_continued_fraction(1.0 - x, b, a) / b


def _beta_continued_fraction(x: float, a: float, b: float) -> float:
    maximum_iterations = 200
    epsilon = 3e-14
    minimum = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / max(abs(d), minimum) * (1.0 if d >= 0.0 else -1.0)
    result = d
    for index in range(1, maximum_iterations + 1):
        even = 2 * index
        numerator = index * (b - index) * x / ((qam + even) * (a + even))
        d = 1.0 + numerator * d
        d = minimum if abs(d) < minimum else d
        c = 1.0 + numerator / c
        c = minimum if abs(c) < minimum else c
        d = 1.0 / d
        result *= d * c

        numerator = -(a + index) * (qab + index) * x / ((a + even) * (qap + even))
        d = 1.0 + numerator * d
        d = minimum if abs(d) < minimum else d
        c = 1.0 + numerator / c
        c = minimum if abs(c) < minimum else c
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            return result
    raise ArithmeticError("incomplete beta continued fraction did not converge")
