"""Pre-inference paired-design power analysis bound to frozen PIVOT_EXP_A."""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any


def build_power_analysis(
    frozen_predictions: str | Path,
    *,
    planned_scenes: int,
    superiority_sesoi: float,
    equivalence_margin: float,
    alpha: float,
    target_power: float,
) -> dict[str, Any]:
    rows = _read_gzip_jsonl(Path(frozen_predictions))
    analogs = {
        "E1_semantic_rescue": ("A2", "complete_evidence", "image_only", alpha),
        "E2_relation_under_oracle": (
            "A2",
            "relation_evidence",
            "image_only",
            alpha / 2.0,
        ),
        "E3_json_minus_triples": ("A3", "json", "triples", alpha / 2.0),
    }
    estimates: dict[str, Any] = {}
    for name, (experiment, left, right, test_alpha) in analogs.items():
        pairs = _paired(rows, experiment=experiment, left=left, right=right)
        counts = Counter(pairs)
        discordance = (counts[(1, 0)] + counts[(0, 1)]) / len(pairs)
        frozen_effect = sum(a - b for a, b in pairs) / len(pairs)
        superiority_power = paired_normal_power(
            planned_scenes,
            effect=superiority_sesoi,
            discordance=discordance,
            alpha=test_alpha,
        )
        equivalence_power = tost_equivalence_power(
            planned_scenes,
            margin=equivalence_margin,
            discordance=discordance,
            alpha=alpha,
        )
        estimates[name] = {
            "frozen_analog": {
                "experiment": experiment,
                "left": left,
                "right": right,
                "n": len(pairs),
                "paired_counts": {f"{a}{b}": counts[(a, b)] for a in (0, 1) for b in (0, 1)},
                "effect": frozen_effect,
                "discordance": discordance,
            },
            "planned_n_scenes": planned_scenes,
            "test_alpha": test_alpha,
            "superiority_power_at_sesoi": superiority_power,
            "tost_power_if_true_effect_zero": equivalence_power,
            "superiority_power_passed": superiority_power >= target_power,
            "equivalence_power_passed": equivalence_power >= target_power,
        }
    return {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A2",
        "status": "PASS"
        if all(
            value["superiority_power_passed"] and value["equivalence_power_passed"]
            for value in estimates.values()
        )
        else "FAIL",
        "computed_before_inference": True,
        "planned_primary_scenes": planned_scenes,
        "superiority_sesoi": superiority_sesoi,
        "equivalence_margin": equivalence_margin,
        "alpha": alpha,
        "target_power": target_power,
        "multiplicity": {
            "E1": "gated at alpha=0.05",
            "E2_E3": "Holm family; worst-case planning alpha=0.025",
        },
        "rationale": (
            "A 0.10 absolute-accuracy threshold was fixed before inference because PIVOT_EXP_A "
            "effects were 0.185-0.415 and only residual effects large enough to motivate an "
            "independent-backbone replication are mechanism-defining."
        ),
        "hop_policy": (
            "Power is for the pooled 450-scene primary estimands. Depth-2 and depth-3 analyses "
            "are preregistered direction/seed replication gates, not separately powered tests."
        ),
        "estimands": estimates,
    }


def paired_normal_power(
    n: int, *, effect: float, discordance: float, alpha: float
) -> float:
    if n <= 0 or not 0 < alpha < 1 or discordance <= effect**2:
        raise ValueError("invalid paired-power inputs")
    normal = NormalDist()
    standard_error = math.sqrt((discordance - effect**2) / n)
    noncentrality = abs(effect) / standard_error
    critical = normal.inv_cdf(1.0 - alpha / 2.0)
    return normal.cdf(noncentrality - critical) + normal.cdf(-noncentrality - critical)


def tost_equivalence_power(
    n: int, *, margin: float, discordance: float, alpha: float
) -> float:
    if n <= 0 or margin <= 0 or discordance <= 0:
        raise ValueError("invalid TOST-power inputs")
    normal = NormalDist()
    standard_error = math.sqrt(discordance / n)
    critical = normal.inv_cdf(1.0 - alpha)
    bound = margin / standard_error - critical
    return max(0.0, min(1.0, 2.0 * normal.cdf(bound) - 1.0))


def _paired(
    rows: list[dict[str, Any]], *, experiment: str, left: str, right: str
) -> list[tuple[int, int]]:
    grouped: dict[tuple[int, str], dict[str, int]] = defaultdict(dict)
    for row in rows:
        if row["experiment"] == experiment:
            grouped[(int(row["seed"]), str(row["source_scene_id"]))][
                str(row["condition"])
            ] = int(bool(row["correct"]))
    pairs = [(values[left], values[right]) for values in grouped.values()]
    if not pairs:
        raise ValueError(f"frozen analog has no pairs: {experiment} {left} {right}")
    return pairs


def _read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


__all__ = ["build_power_analysis", "paired_normal_power", "tost_equivalence_power"]

