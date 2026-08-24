"""Scene-paired, choice-balanced trial construction for PIVOT_EXP_A3."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets.records import SceneRecord
from recoalign.models.vlm.base import PreparedInput

from .answer_contract import PRIMARY_COMPLETION_PREFIX, REGISTERED_CHOICE_IDS
from .scaffold_comprehension import (
    MANIPULATIONS,
    ConstructScaffold,
    build_scaffold_pair,
    serialize_scaffold,
)

TASKS = ("shape", "color", "object_identity", "entity_attribute_binding")
IMAGE_CONTEXTS = ("neutral_image", "original_scene_image")
RESPONSE_METHODS = ("forced_choice", "free_generation")


@dataclass(frozen=True)
class ConstructValidityTrial:
    seed: int
    scene_id: str
    construct: str
    manipulation: str
    evidence_truth: str
    image_context: str
    response_method: str
    task: str
    question: str
    choices: tuple[tuple[str, str], ...]
    scene_truth_answer: str
    declared_answer: str
    scoring_target: str
    correct_choice_id: str
    scene_truth_choice_id: str
    declared_choice_id: str
    prompt: str
    image: str
    scaffold_sha256: str
    metadata: dict[str, Any]

    @property
    def key(self) -> str:
        components = (
            "PIVOT_EXP_A3",
            str(self.seed),
            self.scene_id,
            self.construct,
            self.manipulation,
            self.evidence_truth,
            self.image_context,
            self.response_method,
            self.task,
        )
        return ":".join(components)

    def prepared_input(self) -> PreparedInput:
        token_count = len(self.prompt.split())
        return PreparedInput(
            prompt=self.prompt,
            image=self.image,
            condition=f"{self.manipulation}_{self.evidence_truth}_{self.image_context}",
            setting="construct_validity",
            input_tokens=token_count,
            evidence_tokens=token_count,
            semantic_units=4,
            relation_count=0,
            semantic_facts_sha256=self.scaffold_sha256,
            serialization=self.manipulation,
            padding_units=0,
            token_match_delta=0,
            prompt_protocol_id="pivot-exp-a3-answer-contract-v2",
            prompt_protocol_sha256=None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "study_id": "PIVOT_EXP_A3",
            "trial_key": self.key,
            "seed": self.seed,
            "scene_id": self.scene_id,
            "construct": self.construct,
            "manipulation": self.manipulation,
            "evidence_truth": self.evidence_truth,
            "image_context": self.image_context,
            "response_method": self.response_method,
            "task": self.task,
            "question": self.question,
            "choices": {key: value for key, value in self.choices},
            "scene_truth_answer": self.scene_truth_answer,
            "declared_answer": self.declared_answer,
            "scoring_target": self.scoring_target,
            "correct_choice_id": self.correct_choice_id,
            "scene_truth_choice_id": self.scene_truth_choice_id,
            "declared_choice_id": self.declared_choice_id,
            "prompt": self.prompt,
            "prompt_sha256": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(),
            "image": self.image,
            "scaffold_sha256": self.scaffold_sha256,
            "metadata": self.metadata,
        }


def build_validation_trials(
    records: list[SceneRecord],
    *,
    seed: int,
    neutral_image: str | Path,
    m2_images: dict[tuple[str, str, str], str] | None = None,
    choice_salt: str = "pivot-exp-a3-choice-order-v1",
) -> list[ConstructValidityTrial]:
    ordered_records = sorted(records, key=lambda record: record.scene_id)
    if len(ordered_records) % 4:
        raise ValueError("validation scene count must be divisible by four for exact balance")
    if any(len(record.objects) != 4 for record in ordered_records):
        raise ValueError("every construct-validity scene must contain exactly four objects")
    if len({record.scene_id for record in ordered_records}) != len(ordered_records):
        raise ValueError("construct-validity scene IDs must be unique")
    trials: list[ConstructValidityTrial] = []
    for manipulation in MANIPULATIONS:
        scaffold_pairs = {
            record.scene_id: build_scaffold_pair(record, manipulation, seed=seed)
            for record in ordered_records
        }
        semantic_schedules = {
            task: _position_schedule(
                ordered_records,
                seed=seed,
                task=task,
                manipulation=manipulation,
                choice_salt=choice_salt,
            )
            for task in TASKS
        }
        contract_schedule = _position_schedule(
            ordered_records,
            seed=seed,
            task="answer_contract_comprehension",
            manipulation=manipulation,
            choice_salt=choice_salt,
        )
        for record in ordered_records:
            oracle, corrupted = scaffold_pairs[record.scene_id]
            for scaffold in (oracle, corrupted):
                for task in TASKS:
                    truth_position, declared_position = semantic_schedules[task][record.scene_id]
                    trials.extend(
                        _semantic_trials(
                            record,
                            scaffold,
                            seed=seed,
                            task=task,
                            truth_position=truth_position,
                            declared_position=declared_position,
                            neutral_image=neutral_image,
                            m2_images=m2_images,
                            choice_salt=choice_salt,
                        )
                    )
                truth_position, _ = contract_schedule[record.scene_id]
                trials.extend(
                    _contract_trials(
                        record,
                        scaffold,
                        seed=seed,
                        target_position=truth_position,
                        neutral_image=neutral_image,
                        m2_images=m2_images,
                        choice_salt=choice_salt,
                    )
                )
    if len({trial.key for trial in trials}) != len(trials):
        raise ValueError("construct-validity trial keys are not unique")
    return sorted(trials, key=lambda trial: trial.key)


def _semantic_trials(
    record: SceneRecord,
    scaffold: ConstructScaffold,
    *,
    seed: int,
    task: str,
    truth_position: int,
    declared_position: int,
    neutral_image: str | Path,
    m2_images: dict[tuple[str, str, str], str] | None,
    choice_salt: str,
) -> list[ConstructValidityTrial]:
    scene_answer, declared_answer, question, alternatives = _task_content(record, scaffold, task)
    choices = _ordered_choices(
        alternatives,
        scene_answer=scene_answer,
        declared_answer=declared_answer,
        scene_position=truth_position,
        declared_position=declared_position,
        randomization_key=(
            seed,
            record.scene_id,
            scaffold.manipulation,
            scaffold.evidence_truth,
            task,
        ),
        choice_salt=choice_salt,
    )
    scene_choice = _choice_id_for(choices, scene_answer)
    declared_choice = _choice_id_for(choices, declared_answer)
    trials = []
    for image_context in IMAGE_CONTEXTS:
        response_methods = ["forced_choice"]
        if scaffold.manipulation in {"M1", "M2"} and scaffold.evidence_truth == "oracle":
            response_methods.append("free_generation")
        for response_method in response_methods:
            scoring_target = (
                "active_scaffold_declaration"
                if image_context == "neutral_image"
                else "scene_truth"
            )
            correct_choice = declared_choice if image_context == "neutral_image" else scene_choice
            trials.append(
                _trial(
                    record,
                    scaffold,
                    seed=seed,
                    construct="CV2" if image_context == "neutral_image" else "CV3",
                    image_context=image_context,
                    response_method=response_method,
                    task=task,
                    question=question,
                    choices=choices,
                    scene_answer=scene_answer,
                    declared_answer=declared_answer,
                    scoring_target=scoring_target,
                    correct_choice=correct_choice,
                    scene_choice=scene_choice,
                    declared_choice=declared_choice,
                    neutral_image=neutral_image,
                    m2_images=m2_images,
                )
            )
    return trials


def _contract_trials(
    record: SceneRecord,
    scaffold: ConstructScaffold,
    *,
    seed: int,
    target_position: int,
    neutral_image: str | Path,
    m2_images: dict[tuple[str, str, str], str] | None,
    choice_salt: str,
) -> list[ConstructValidityTrial]:
    aliases = tuple(binding.entity_id for binding in scaffold.bindings)
    target = scaffold.target_entity_id
    choices = _ordered_choices(
        aliases,
        scene_answer=target,
        declared_answer=target,
        scene_position=target_position,
        declared_position=target_position,
        randomization_key=(
            seed,
            record.scene_id,
            scaffold.manipulation,
            scaffold.evidence_truth,
            "answer_contract_comprehension",
        ),
        choice_salt=choice_salt,
    )
    choice_id = _choice_id_for(choices, target)
    methods = ["forced_choice"]
    if scaffold.manipulation in {"M1", "M2"} and scaffold.evidence_truth == "oracle":
        methods.append("free_generation")
    return [
        _trial(
            record,
            scaffold,
            seed=seed,
            construct="CV1",
            image_context="neutral_image",
            response_method=method,
            task="answer_contract_comprehension",
            question=f"Which option text is exactly {target}?",
            choices=choices,
            scene_answer=target,
            declared_answer=target,
            scoring_target="active_scaffold_declaration",
            correct_choice=choice_id,
            scene_choice=choice_id,
            declared_choice=choice_id,
            neutral_image=neutral_image,
            m2_images=m2_images,
        )
        for method in methods
    ]


def _trial(
    record: SceneRecord,
    scaffold: ConstructScaffold,
    *,
    seed: int,
    construct: str,
    image_context: str,
    response_method: str,
    task: str,
    question: str,
    choices: tuple[tuple[str, str], ...],
    scene_answer: str,
    declared_answer: str,
    scoring_target: str,
    correct_choice: str,
    scene_choice: str,
    declared_choice: str,
    neutral_image: str | Path,
    m2_images: dict[tuple[str, str, str], str] | None,
) -> ConstructValidityTrial:
    scaffold_text = serialize_scaffold(scaffold)
    prompt = _render_prompt(
        scaffold_text,
        question=question,
        choices=choices,
        response_method=response_method,
    )
    if scaffold.manipulation == "M2":
        if m2_images is None:
            raise ValueError("M2 trial construction requires frozen composed legend images")
        image = m2_images[(record.scene_id, scaffold.evidence_truth, image_context)]
    else:
        image = str(neutral_image) if image_context == "neutral_image" else record.image
    scaffold_digest = hashlib.sha256(
        json.dumps(scaffold.canonical_facts, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ConstructValidityTrial(
        seed=seed,
        scene_id=record.scene_id,
        construct=construct,
        manipulation=scaffold.manipulation,
        evidence_truth=scaffold.evidence_truth,
        image_context=image_context,
        response_method=response_method,
        task=task,
        question=question,
        choices=choices,
        scene_truth_answer=scene_answer,
        declared_answer=declared_answer,
        scoring_target=scoring_target,
        correct_choice_id=correct_choice,
        scene_truth_choice_id=scene_choice,
        declared_choice_id=declared_choice,
        prompt=prompt,
        image=image,
        scaffold_sha256=scaffold_digest,
        metadata={
            "row_order": list(scaffold.row_order),
            "corruption_shift": scaffold.corruption_shift,
            "choice_ids": list(REGISTERED_CHOICE_IDS),
            "relation_fact_count": 0,
            "semantic_fact_count": len(scaffold.bindings),
            "oracle_scene_facts_consistent": scaffold.evidence_truth == "oracle",
            "all_corrupted_bindings_false": scaffold.evidence_truth == "corrupted",
            "contains_query_role": False,
            "contains_reasoning_chain": False,
            "contains_final_answer_field": False,
        },
    )


def _task_content(
    record: SceneRecord, scaffold: ConstructScaffold, task: str
) -> tuple[str, str, str, tuple[str, ...]]:
    del record
    active = scaffold.by_entity()
    target = scaffold.target_entity_id
    declared = active[target]
    aliases = tuple(binding.entity_id for binding in scaffold.bindings)
    target_object = next(
        object_id for object_id, entity in scaffold.alias_by_object_id.items() if entity == target
    )
    scene_binding = next(
        binding for binding in scaffold.bindings if binding.source_object_id == target_object
    )
    if scaffold.evidence_truth == "corrupted":
        # Recover immutable scene truth from the binding whose source object is the target object.
        scene_shape = scene_binding.shape
        scene_color = scene_binding.color
    else:
        scene_shape = declared.shape
        scene_color = declared.color
    # The corrupt scaffold contains the full original attribute multiset, so source-object lookup
    # above recovers scene truth without consulting a model response.
    if task == "shape":
        alternatives = tuple(binding.shape for binding in scaffold.bindings)
        return scene_shape, declared.shape, f"Which shape belongs to {target}?", alternatives
    if task == "color":
        alternatives = tuple(binding.color for binding in scaffold.bindings)
        return scene_color, declared.color, f"Which color belongs to {target}?", alternatives
    if task == "entity_attribute_binding":
        alternatives = tuple(binding.description for binding in scaffold.bindings)
        return (
            f"{scene_color} {scene_shape}",
            declared.description,
            f"Which description belongs to {target}?",
            alternatives,
        )
    if task == "object_identity":
        scene_description = (scene_shape, scene_color)
        declared_entity = next(
            entity
            for entity, binding in active.items()
            if (binding.shape, binding.color) == scene_description
        )
        return (
            target,
            declared_entity,
            f"Which entity is the {scene_color} {scene_shape}?",
            aliases,
        )
    raise ValueError(task)


def _position_schedule(
    records: list[SceneRecord],
    *,
    seed: int,
    task: str,
    manipulation: str,
    choice_salt: str,
) -> dict[str, tuple[int, int]]:
    randomized = sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"{seed}|{record.scene_id}|{task}|{manipulation}|{choice_salt}".encode()
        ).hexdigest(),
    )
    schedule = {}
    for index, record in enumerate(randomized):
        scene_position = index % 4 + 1
        offset = 1 + (index // 4) % 3
        declared_position = (scene_position - 1 + offset) % 4 + 1
        schedule[record.scene_id] = (scene_position, declared_position)
    return schedule


def _ordered_choices(
    alternatives: tuple[str, ...],
    *,
    scene_answer: str,
    declared_answer: str,
    scene_position: int,
    declared_position: int,
    randomization_key: tuple[Any, ...],
    choice_salt: str,
) -> tuple[tuple[str, str], ...]:
    if len(alternatives) != 4 or len(set(alternatives)) != 4:
        raise ValueError(
            f"registered choices must be four unique scene alternatives: {alternatives}"
        )
    if scene_answer not in alternatives or declared_answer not in alternatives:
        raise ValueError("scene and declared answers must be registered alternatives")
    slots: list[str | None] = [None, None, None, None]
    slots[scene_position - 1] = scene_answer
    if declared_answer != scene_answer:
        if declared_position == scene_position:
            raise ValueError("distinct scene and declared answers require distinct positions")
        slots[declared_position - 1] = declared_answer
    remaining = [value for value in alternatives if value not in {scene_answer, declared_answer}]
    digest = hashlib.sha256(
        ("|".join(str(value) for value in randomization_key) + f"|{choice_salt}").encode("utf-8")
    ).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(remaining)
    iterator = iter(remaining)
    for index, value in enumerate(slots):
        if value is None:
            slots[index] = next(iterator)
    return tuple(zip(REGISTERED_CHOICE_IDS, (str(value) for value in slots), strict=True))


def _choice_id_for(choices: tuple[tuple[str, str], ...], answer: str) -> str:
    matches = [choice_id for choice_id, value in choices if value == answer]
    if len(matches) != 1:
        raise ValueError(f"answer must map to exactly one option: {answer}")
    return matches[0]


def _render_prompt(
    scaffold: str,
    *,
    question: str,
    choices: tuple[tuple[str, str], ...],
    response_method: str,
) -> str:
    options = "\n".join(f"{choice_id} | {value}" for choice_id, value in choices)
    common = (
        f"SEMANTIC_SCAFFOLD\n{scaffold}\n\nQUESTION\n{question}\n\n"
        f"OPTIONS\n{options}\n\nANSWER_CONTRACT=answer-contract-v2\n"
    )
    if response_method == "forced_choice":
        return common + "Select exactly one registered option ID.\n" + PRIMARY_COMPLETION_PREFIX
    if response_method == "free_generation":
        return common + "Return exactly one line in the form FINAL_CHOICE=<1|2|3|4>."
    raise ValueError(response_method)


__all__ = [
    "ConstructValidityTrial",
    "IMAGE_CONTEXTS",
    "RESPONSE_METHODS",
    "TASKS",
    "build_validation_trials",
]
