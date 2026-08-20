"""Registered ablation hypotheses and interpretation rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AblationSpec:
    identifier: str
    family: str
    hypothesis: str
    variable_changed: str
    expected_outcome: str
    interpretation: str
    required_seeds: int = 3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ABLATION_SPECS: tuple[AblationSpec, ...] = (
    AblationSpec(
        "A1_FULL_RECOALIGN", "architecture", "The complete interface improves reasoning.",
        "learned structure-token encoder enabled",
        "reference method defines the comparison anchor.",
        "Use only as the paired full-model reference; it is not an ablation.",
    ),
    AblationSpec(
        "A2_NO_STRUCTURE_INTERFACE", "architecture", "Structure tokens are necessary.",
        "remove structure-token module with matched training budget", "accuracy decreases.",
        "A drop attributes gains to the interface rather than generic training.",
    ),
    AblationSpec(
        "A3_RANDOM_STRUCTURE_TOKENS", "architecture", "Learned structure is necessary.",
        "replace inferred tokens by seeded random tokens", "accuracy and probes decrease.",
        "A drop rejects the extra-token-capacity explanation.",
    ),
    AblationSpec(
        "A4_FIXED_GRAPH_ENCODER",
        "architecture",
        "A learned interface differs from fixed graph injection.",
        "replace learned slots with a fixed graph encoder", "performance/mechanism differs.",
        "Separates latent interface learning from handcrafted graph knowledge.",
    ),
    AblationSpec(
        "L1_NO_SEMANTIC_PRESERVATION", "loss", "Semantic preservation retains visual capability.",
        "semantic loss weight set to zero", "semantic probes or capability may decline.",
        "Attributes preservation to the semantic objective.",
    ),
    AblationSpec(
        "L2_NO_STRUCTURAL_CONSISTENCY", "loss", "Structural supervision organizes relation tokens.",
        "structural loss weight set to zero", "structure probes and OOD may decline.",
        "Measures the role of graph labels as training signal only.",
    ),
    AblationSpec(
        "L3_NO_REASONING_ALIGNMENT", "loss", "Reasoning alignment connects tokens to answers.",
        "reasoning loss weight set to zero", "QA accuracy declines.",
        "Attributes downstream use to reasoning alignment.",
    ),
    AblationSpec(
        "S1_FULL_GRAPH_SUPERVISION", "supervision", "Full graph labels teach the interface.",
        "object/attribute/relation/composition labels", "reference supervision condition.",
        "Anchor for weaker supervision controls.",
    ),
    AblationSpec(
        "S2_WEAK_STRUCTURAL_SUPERVISION",
        "supervision",
        "The interface does not require complete graphs.",
        "relation-only or partial structural labels", "smaller but nonzero structure signal.",
        "Tests whether the method copies a full graph annotation.",
    ),
    AblationSpec(
        "S3_NO_GRAPH_SUPERVISION", "supervision", "Reasoning signal can learn usable structure.",
        "remove graph labels; retain QA supervision",
        "probe may weaken but inference remains graph-free.",
        "Tests whether graph supervision is necessary rather than merely helpful.",
        5,
    ),
    AblationSpec(
        "P1_PARAMETER_MATCHED", "capacity", "Extra random capacity cannot reproduce the gain.",
        "add equal parameter count without structure computation", "ReCoAlign remains better.",
        "Rejects parameter-count attribution.",
        5,
    ),
    AblationSpec(
        "I1_REMOVE_TOKENS", "causal", "Structure tokens have a causal effect.",
        "zero structure tokens at inference", "accuracy decreases.",
        "Necessary-token intervention.",
        5,
    ),
    AblationSpec(
        "I2_SHUFFLE_TOKENS", "causal", "Slot organization matters.",
        "permute tokens within each sample", "accuracy decreases or predictions change.",
        "Tests slot-specific causal use.",
        5,
    ),
    AblationSpec(
        "I3_REPLACE_TOKENS", "causal", "Tokens carry sample-specific evidence.",
        "replace tokens with another sample's tokens", "cross-sample errors increase.",
        "Tests causal content rather than token presence.",
        5,
    ),
)


def load_mechanistic_registry(
    path: str | Path = "configs/ablations/mechanistic_registry.yaml",
) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("ablations"), list):
        raise ValueError("mechanistic registry requires an ablations list")
    return payload


def validate_mechanistic_registry(
    path: str | Path = "configs/ablations/mechanistic_registry.yaml",
) -> dict[str, Any]:
    payload = load_mechanistic_registry(path)
    rows = payload["ablations"]
    expected = {spec.identifier for spec in ABLATION_SPECS}
    observed = {str(row.get("id")) for row in rows if isinstance(row, dict)}
    if observed != expected:
        raise ValueError(
            "mechanistic registry IDs differ from implementation: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    for row in rows:
        if not isinstance(row, dict) or not all(
            isinstance(row.get(field), str) and row[field].strip()
            for field in (
                "id",
                "family",
                "hypothesis",
                "variable_changed",
                "expected_outcome",
                "interpretation",
            )
        ):
            raise ValueError(
                "each ablation must register hypothesis, variable, outcome, interpretation"
            )
        if int(row.get("required_seeds", 3)) < 3:
            raise ValueError("mechanistic ablations require at least three seeds")
    return {
        "valid": True,
        "ablation_count": len(rows),
        "families": sorted({str(row["family"]) for row in rows}),
        "no_hidden_ablation": True,
    }


__all__ = [
    "ABLATION_SPECS",
    "AblationSpec",
    "load_mechanistic_registry",
    "validate_mechanistic_registry",
]
