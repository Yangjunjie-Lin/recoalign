"""Frozen construct-validity decision engine for PIVOT_EXP_A3."""

from __future__ import annotations

from typing import Any

ALLOWED_OUTCOMES = {
    "CONSTRUCT_VALIDITY_GO_TEXT",
    "CONSTRUCT_VALIDITY_GO_VISUAL_LEGEND",
    "ANSWER_CONTRACT_FAILURE",
    "SEMANTIC_MANIPULATION_FAILURE",
    "IMAGE_INTERFERENCE_DIAGNOSIS",
    "NO-GO_CONSTRUCT",
    "INCONCLUSIVE",
}


def adjudicate_construct_validity(
    analysis: dict[str, Any], *, integrity_passed: bool
) -> dict[str, Any]:
    if not integrity_passed or len(analysis.get("completed_seeds", [])) != 5:
        return _decision(
            "INCONCLUSIVE",
            selected=None,
            reason=(
                "A registered execution, sample, hash, crop, seed, or trial integrity "
                "gate failed."
            ),
        )
    candidates = analysis["candidates"]
    answer_contract_passed = all(
        bool(candidates[name]["gates"]["A"])
        and bool(candidates[name]["gates"]["B"])
        for name in ("M1", "M2")
    )
    if not answer_contract_passed:
        return _decision(
            "ANSWER_CONTRACT_FAILURE",
            selected=None,
            reason=(
                "Complete hash-valid data failed primary or secondary answer-contract "
                "integrity."
            ),
        )
    if bool(candidates["M1"]["passed_all_gates"]):
        return _decision(
            "CONSTRUCT_VALIDITY_GO_TEXT",
            selected="M1",
            reason="M1 passed every task-specific Gate A-F; minimum intervention selects text.",
            allow_a4=True,
        )
    if bool(candidates["M2"]["passed_all_gates"]):
        return _decision(
            "CONSTRUCT_VALIDITY_GO_VISUAL_LEGEND",
            selected="M2",
            reason="M1 failed and M2 passed every task-specific Gate A-F.",
            allow_a4=True,
        )
    neutral_pass = any(bool(candidates[name]["gates"]["C"]) for name in ("M1", "M2"))
    image_failure = all(
        not (
            bool(candidates[name]["gates"]["D"])
            and bool(candidates[name]["gates"]["F"])
        )
        for name in ("M1", "M2")
        if bool(candidates[name]["gates"]["C"])
    )
    if neutral_pass and image_failure:
        return _decision(
            "IMAGE_INTERFERENCE_DIAGNOSIS",
            selected=None,
            reason=(
                "Declared-fact comprehension passed under neutral input but original-image "
                "sufficiency or equivalence failed."
            ),
        )
    primitive_failure = all(
        not all(bool(candidates[name]["gates"][gate]) for gate in ("C", "D", "E"))
        for name in ("M1", "M2")
    )
    if primitive_failure:
        return _decision(
            "SEMANTIC_MANIPULATION_FAILURE",
            selected=None,
            reason=(
                "The answer contract passed, but neither M1 nor M2 established all four "
                "primitive semantic constructs."
            ),
        )
    return _decision(
        "NO-GO_CONSTRUCT",
        selected=None,
        reason=(
            "The complete failure pattern matches no single preregistered integrity or "
            "image-interference diagnosis."
        ),
    )


def _decision(
    outcome: str,
    *,
    selected: str | None,
    reason: str,
    allow_a4: bool = False,
) -> dict[str, Any]:
    if outcome not in ALLOWED_OUTCOMES:
        raise ValueError(f"unregistered construct-validity outcome: {outcome}")
    return {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A3",
        "outcome": outcome,
        "selected_manipulation": selected,
        "reason": reason,
        "authorization": {
            "PIVOT_EXP_A4_preregistration_allowed": allow_a4,
            "independent_backbone_replication_allowed": False,
            "model_development_allowed": False,
            "paper_writing_allowed": False,
        },
    }


__all__ = ["ALLOWED_OUTCOMES", "adjudicate_construct_validity"]
