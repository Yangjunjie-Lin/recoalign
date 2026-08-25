from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from recoalign.construct_validity.answer_contract import (
    REGISTERED_CHOICE_IDS,
    parse_final_choice,
    validate_parser_contract,
)
from recoalign.construct_validity.choice_scoring import (
    score_choices,
    validate_tokenizer_contract,
)
from recoalign.construct_validity.decision import (
    ALLOWED_OUTCOMES,
    adjudicate_construct_validity,
)
from recoalign.construct_validity.integrity import (
    A2_FROZEN_SHA256,
    audit_legacy_parse_failures,
    validate_a2_frozen_assets,
)
from recoalign.construct_validity.scaffold_comprehension import (
    build_scaffold_pair,
    serialize_scaffold,
    validate_scaffold_pair,
)
from recoalign.construct_validity.statistics import holm_correction, paired_tost
from recoalign.construct_validity.trial_builder import TASKS, build_validation_trials
from recoalign.models.vlm.base import PreparedInput
from recoalign.synthetic_world import GeneratorConfig, SyntheticWorldGenerator

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(
    (ROOT / "research/construct_validity/PIVOT_EXP_A3/config.yaml").read_text(encoding="utf-8")
)


class _FakeTokenizer:
    token_ids = {"1": 101, "2": 102, "3": 103, "4": 104}

    def __call__(self, value: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        del add_special_tokens
        prefix = "FINAL_CHOICE="
        if value in self.token_ids:
            return {"input_ids": [self.token_ids[value]]}
        if value.startswith(prefix) and value[len(prefix) :] in self.token_ids:
            return {"input_ids": [10, 11, self.token_ids[value[len(prefix) :]]]}
        if value == prefix:
            return {"input_ids": [10, 11]}
        if value.startswith(" ") and value[1:] in self.token_ids:
            return {"input_ids": [99, self.token_ids[value[1:]]]}
        return {"input_ids": [500]}


class _FakeLikelihoodBackend:
    def score_choice_log_likelihoods(
        self, prepared_input: PreparedInput, choice_ids: tuple[str, ...]
    ) -> dict[str, float]:
        assert prepared_input.prompt.endswith("FINAL_CHOICE=")
        return {choice: -abs(int(choice) - 3) for choice in choice_ids}


def _prepared() -> PreparedInput:
    return PreparedInput(
        prompt="question\nFINAL_CHOICE=",
        image="neutral.png",
        condition="test",
        setting="test",
        input_tokens=1,
        evidence_tokens=1,
        semantic_units=0,
        relation_count=0,
        semantic_facts_sha256=None,
        serialization="test",
        padding_units=0,
        token_match_delta=0,
    )


def _records(count: int = 100, seed: int = 20260901) -> list[Any]:
    generator = SyntheticWorldGenerator(GeneratorConfig(seed=seed, write_images=False))
    candidates = generator.generate(count * 12, seed=seed, split="construct_validity")
    return sorted(
        (record for record in candidates if len(record.objects) == 4),
        key=lambda record: record.scene_id,
    )[:count]


def _fake_m2_images(records: list[Any]) -> dict[tuple[str, str, str], str]:
    return {
        (record.scene_id, truth, context): "frozen_m2_composite.png"
        for record in records
        for truth in ("oracle", "corrupted")
        for context in ("neutral_image", "original_scene_image")
    }


def _analysis(
    *,
    m1: dict[str, bool] | None = None,
    m2: dict[str, bool] | None = None,
    gate_a: bool = True,
) -> dict[str, Any]:
    default = {gate: False for gate in "ABCDEF"}

    def candidate(gates: dict[str, bool] | None) -> dict[str, Any]:
        values = {**default, **(gates or {})}
        return {
            "gates": values,
            "passed_all_gates": all(values.values()),
        }

    return {
        "completed_seeds": [1, 2, 3, 4, 5],
        "gate_A": {"passed": gate_a},
        "candidates": {"M1": candidate(m1), "M2": candidate(m2)},
    }


def test_a2_frozen_assets_match_registered_sha256() -> None:
    report = validate_a2_frozen_assets(ROOT)
    assert report["passed"] is True
    assert set(report["assets"]) == set(A2_FROZEN_SHA256)
    assert all(value["matches"] for value in report["assets"].values())


def test_legacy_137_rows_are_audited_without_rescoring(tmp_path: Path) -> None:
    source = ROOT / "research/causal_separation/PIVOT_EXP_A2/results/predictions.jsonl.gz"
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    summary = audit_legacy_parse_failures(ROOT, tmp_path)
    after = hashlib.sha256(source.read_bytes()).hexdigest()
    assert before == after == A2_FROZEN_SHA256[source.relative_to(ROOT).as_posix()]
    assert summary["selection"]["selected_count"] == 137
    assert summary["counts"]["by_raw_output_pattern"] == {
        "hallucinated_choice": 19,
        "multiple_choices": 118,
    }
    assert summary["research_integrity"]["PIVOT_EXP_A2_predictions_rescored"] is False


def test_answer_contract_adversarial_inventory_passes() -> None:
    report = validate_parser_contract()
    assert report["passed"] is True
    assert len(report["cases"]) >= 12


@pytest.mark.parametrize(
    ("raw", "valid"),
    [
        ("FINAL_CHOICE=1", True),
        ("final_choice=1", False),
        ("FINAL_CHOICE＝1", False),
        ("FINAL_CHOICE=1\nFINAL_CHOICE=1", False),
        ("Because E1 is red, FINAL_CHOICE=1", False),
        ("FINAL_CHOICE=14", False),
    ],
)
def test_frozen_parser_does_not_accept_aliases_or_substrings(raw: str, valid: bool) -> None:
    assert parse_final_choice(raw).valid is valid


def test_tokenizer_contract_freezes_no_leading_space_boundary() -> None:
    report = validate_tokenizer_contract(_FakeTokenizer())
    assert report["passed"] is True
    assert report["unique_token_ids"] is True
    assert all(row["single_token"] for row in report["options"])
    assert all(row["leading_space_sensitive"] for row in report["options"])
    assert all(row["primary_method_uses_leading_space"] is False for row in report["options"])


def test_forced_choice_argmax_has_no_ground_truth_argument() -> None:
    result = score_choices(_prepared(), REGISTERED_CHOICE_IDS, backend=_FakeLikelihoodBackend())
    assert result.predicted_choice_id == "3"
    assert result.valid_measurement is True
    assert set(result.log_likelihoods) == set(REGISTERED_CHOICE_IDS)


@pytest.mark.parametrize("manipulation", ["M0", "M1", "M2"])
def test_oracle_corrupted_scaffolds_are_complete_false_derangements(
    manipulation: str,
) -> None:
    record = _records(1)[0]
    oracle, corrupted = build_scaffold_pair(record, manipulation, seed=20260901)
    assertions = validate_scaffold_pair(oracle, corrupted)
    assert all(assertions.values())
    assert len(oracle.bindings) == len(corrupted.bindings) == 4
    assert oracle.row_order == corrupted.row_order
    assert all(
        (oracle.by_entity()[entity].shape, oracle.by_entity()[entity].color)
        != (corrupted.by_entity()[entity].shape, corrupted.by_entity()[entity].color)
        for entity in oracle.by_entity()
    )


def test_M1_table_and_M2_prompt_have_no_relation_answer_or_query_role() -> None:
    record = _records(1)[0]
    m1 = serialize_scaffold(build_scaffold_pair(record, "M1", seed=20260901)[0])
    m2 = serialize_scaffold(build_scaffold_pair(record, "M2", seed=20260901)[0])
    assert m1.startswith("ENTITY_TABLE\n")
    assert {line.split(" | ")[0] for line in m1.splitlines()[1:]} == {
        "E1",
        "E2",
        "E3",
        "E4",
    }
    forbidden = ("target_relation", "query_role", "final_answer", "reasoning_chain")
    assert all(value not in m1.casefold() for value in forbidden)
    assert all(value not in m2.casefold() for value in forbidden)


def test_choices_are_scene_alternatives_without_fillers_and_namespaces_do_not_collide() -> None:
    records = _records(4)
    trials = build_validation_trials(
        records,
        seed=20260901,
        neutral_image="neutral.png",
        m2_images=_fake_m2_images(records),
    )
    assert len(trials) == 4 * 72
    for trial in trials:
        choices = dict(trial.choices)
        assert tuple(choices) == REGISTERED_CHOICE_IDS
        assert len(set(choices.values())) == 4
        assert all(not value.startswith(("choice ", "description ")) for value in choices.values())
        entity_values = [value for value in choices.values() if value.startswith(("E", "Entity"))]
        assert all(value not in REGISTERED_CHOICE_IDS for value in entity_values)


def test_correct_positions_are_exactly_balanced_for_scene_and_declared_targets() -> None:
    records = _records(100)
    trials = build_validation_trials(
        records,
        seed=20260901,
        neutral_image="neutral.png",
        m2_images=_fake_m2_images(records),
    )
    expected = Counter({"1": 25, "2": 25, "3": 25, "4": 25})
    for manipulation in ("M0", "M1", "M2"):
        for truth in ("oracle", "corrupted"):
            for task in TASKS:
                cv3 = [
                    trial
                    for trial in trials
                    if trial.manipulation == manipulation
                    and trial.evidence_truth == truth
                    and trial.task == task
                    and trial.construct == "CV3"
                    and trial.response_method == "forced_choice"
                ]
                cv2 = [
                    trial
                    for trial in trials
                    if trial.manipulation == manipulation
                    and trial.evidence_truth == truth
                    and trial.task == task
                    and trial.construct == "CV2"
                    and trial.response_method == "forced_choice"
                ]
                assert Counter(trial.scene_truth_choice_id for trial in cv3) == expected
                assert Counter(trial.declared_choice_id for trial in cv2) == expected


def test_development_and_validation_seeds_and_A2_seeds_are_disjoint() -> None:
    a2 = {20260818, 20260819, 20260820, 20260821, 20260822}
    validation = set(CONFIG["study"]["validation_seeds"])
    development = int(CONFIG["study"]["development_seed"])
    assert not (a2 & validation)
    assert development not in a2 | validation


def test_trial_keys_are_unique_and_counts_match_registration() -> None:
    records = _records(4)
    trials = build_validation_trials(
        records,
        seed=20260901,
        neutral_image="neutral.png",
        m2_images=_fake_m2_images(records),
    )
    assert len({trial.key for trial in trials}) == len(trials)
    assert Counter(trial.response_method for trial in trials) == {
        "forced_choice": 4 * 54,
        "free_generation": 4 * 18,
    }


def test_decision_engine_classifies_registered_outcomes() -> None:
    all_pass = {gate: True for gate in "ABCDEF"}
    answer_pass = {"A": True, "B": True}
    cases = {
        "CONSTRUCT_VALIDITY_GO_TEXT": (_analysis(m1=all_pass, m2=all_pass), True),
        "CONSTRUCT_VALIDITY_GO_VISUAL_LEGEND": (
            _analysis(m1=answer_pass, m2=all_pass),
            True,
        ),
        "ANSWER_CONTRACT_FAILURE": (_analysis(gate_a=False), True),
        "SEMANTIC_MANIPULATION_FAILURE": (
            _analysis(m1=answer_pass, m2=answer_pass),
            True,
        ),
        "IMAGE_INTERFERENCE_DIAGNOSIS": (
            _analysis(
                m1={**answer_pass, "C": True, "D": False, "E": True, "F": False},
                m2=answer_pass,
            ),
            True,
        ),
        "INCONCLUSIVE": (_analysis(m1=all_pass), False),
    }
    no_go = _analysis(m1=all_pass, m2=answer_pass)
    no_go["candidates"]["M1"]["passed_all_gates"] = False
    cases["NO-GO_CONSTRUCT"] = (no_go, True)
    observed = {
        adjudicate_construct_validity(payload, integrity_passed=integrity)["outcome"]
        for payload, integrity in cases.values()
    }
    assert set(cases) <= observed
    assert observed <= ALLOWED_OUTCOMES


def test_task_failure_cannot_be_overridden_by_aggregate() -> None:
    gates = {gate: True for gate in "ABCDEF"}
    gates["D"] = False
    decision = adjudicate_construct_validity(
        _analysis(m1=gates, m2={"A": True, "B": True}), integrity_passed=True
    )
    assert not decision["outcome"].startswith("CONSTRUCT_VALIDITY_GO")


def test_holm_and_tost_synthetic_contracts() -> None:
    holm = holm_correction({"shape": 0.001, "color": 0.02, "binding": 0.2}, alpha=0.05)
    assert holm["shape"]["reject"] is True
    assert holm["binding"]["reject"] is False
    assert paired_tost([0.0] * 100, margin=0.05, alpha=0.05)["equivalent_unadjusted"] is True
    assert paired_tost([0.10] * 100, margin=0.05, alpha=0.05)["equivalent_unadjusted"] is False
