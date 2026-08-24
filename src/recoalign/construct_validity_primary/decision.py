"""Frozen candidate selection and outcome classification for PIVOT_EXP_A3P."""

from __future__ import annotations

from typing import Any

ALLOWED_OUTCOMES = {
    "CONSTRUCT_VALIDITY_GO_TEXT",
    "CONSTRUCT_VALIDITY_GO_VISUAL_LEGEND",
    "PRIMARY_MEASUREMENT_FAILURE",
    "SEMANTIC_MANIPULATION_FAILURE",
    "IMAGE_INTERFERENCE_DIAGNOSIS",
    "NO-GO_CONSTRUCT",
    "INCONCLUSIVE",
}


def adjudicate_primary_construct_validity(
    analysis: dict[str, Any], *, integrity: dict[str, Any]
) -> dict[str, Any]:
    if integrity.get("primary_measurement_failure"):
        return _decision(
            "PRIMARY_MEASUREMENT_FAILURE",
            reason=(
                "The frozen primary scorer did not produce exactly four finite registered "
                "scores for every row."
            ),
        )
    if not integrity.get("passed") or analysis.get("completed_seeds") != [
        20260901,
        20260902,
        20260903,
        20260904,
        20260905,
    ]:
        return _decision(
            "INCONCLUSIVE",
            reason=(
                "A preregistered integrity, completeness, hash, asset, or environment "
                "requirement failed."
            ),
        )
    candidates = analysis["candidates"]
    if bool(candidates["M1"]["passed_all_gates"]):
        return _decision(
            "CONSTRUCT_VALIDITY_GO_TEXT",
            selected="M1",
            reason=(
                "M1 passed Gates A, C, D, E, and F and is selected by the "
                "minimum-intervention principle."
            ),
            allow_a4=True,
        )
    if bool(candidates["M2"]["passed_all_gates"]):
        return _decision(
            "CONSTRUCT_VALIDITY_GO_VISUAL_LEGEND",
            selected="M2",
            reason="M1 failed while M2 passed Gates A, C, D, E, and F.",
            allow_a4=True,
        )
    neutral_pass = any(bool(candidates[name]["gates"]["C"]) for name in ("M1", "M2"))
    image_failure = any(
        bool(candidates[name]["gates"]["C"])
        and (not candidates[name]["gates"]["D"] or not candidates[name]["gates"]["F"])
        for name in ("M1", "M2")
    )
    if neutral_pass and image_failure:
        return _decision(
            "IMAGE_INTERFERENCE_DIAGNOSIS",
            reason=(
                "Neutral scaffold comprehension passed, but original-image sufficiency or "
                "paired equivalence failed."
            ),
        )
    semantic_failure = all(
        not all(bool(candidates[name]["gates"][gate]) for gate in ("C", "D", "E"))
        for name in ("M1", "M2")
    )
    if analysis["gate_A"]["passed"] and semantic_failure:
        return _decision(
            "SEMANTIC_MANIPULATION_FAILURE",
            reason=(
                "Primary measurement remained valid, but neither M1 nor M2 passed all "
                "semantic Gates C, D, and E."
            ),
        )
    return _decision(
        "NO-GO_CONSTRUCT",
        reason=(
            "The valid primary results did not support a candidate and did not match a more "
            "specific preregistered diagnosis."
        ),
    )


def _decision(
    outcome: str,
    *,
    reason: str,
    selected: str | None = None,
    allow_a4: bool = False,
) -> dict[str, Any]:
    if outcome not in ALLOWED_OUTCOMES:
        raise ValueError(f"unregistered PIVOT_EXP_A3P outcome: {outcome}")
    if selected not in {None, "M1", "M2"}:
        raise ValueError("M0 and unregistered combinations cannot be selected")
    return {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A3P",
        "outcome": outcome,
        "selected_manipulation": selected,
        "selection_rule": "minimum-intervention principle",
        "reason": reason,
        "secondary_gate": {
            "status": "RETIRED_BEFORE_VALIDATION",
            "participates_in_scientific_decision": False,
            "source_study": "PIVOT_EXP_A3R",
        },
        "authorization": {
            "PIVOT_EXP_A4_preregistration_allowed": allow_a4,
            "independent_backbone_replication_allowed": False,
            "model_development_allowed": False,
            "paper_writing_allowed": False,
        },
    }


__all__ = ["ALLOWED_OUTCOMES", "adjudicate_primary_construct_validity"]
