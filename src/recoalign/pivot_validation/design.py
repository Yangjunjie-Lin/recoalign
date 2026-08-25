"""Frozen A1/A2/A3 behavioral interventions for PH001 validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from datasets.records import SceneRecord
from recoalign.models.vlm.base import BaseVLM, PreparedInput
from recoalign.synthetic_world.ontology import COLORS, RELATIONS, SHAPES, normalize_relation
from recoalign.synthetic_world.questions import canonical_facts, facts_sha256, object_description

QUESTION_TYPES = ("multi_hop", "relation_reasoning", "object_reasoning", "attribute_reasoning")
A1_TASKS = ("object_shape", "attribute_color", "direct_relation")
A2_CONDITIONS = ("image_only", "object_evidence", "relation_evidence", "complete_evidence")
A3_CONDITIONS = ("natural_language", "triples", "json")
_PADDING_UNIT = " ."

_RELATION_PHRASES = {
    "left": "is to the left of",
    "right": "is to the right of",
    "above": "is above",
    "below": "is below",
    "front": "is in front of",
    "behind": "is behind",
    "near": "is near",
    "far": "is far from",
    "inside": "is inside",
    "contains": "contains",
    "touching": "is touching",
    "holding": "is holding",
}


@dataclass(frozen=True)
class PivotTrial:
    experiment: str
    seed: int
    source_scene_id: str
    trial_id: str
    condition: str
    task: str
    record: SceneRecord
    prepared: PreparedInput
    metadata: dict[str, Any]

    @property
    def key(self) -> str:
        return f"{self.experiment}:{self.seed}:{self.trial_id}:{self.condition}"


def load_source_records(path: str | Path) -> list[SceneRecord]:
    source = Path(path)
    rows = [
        SceneRecord.from_dict(json.loads(line))
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"pivot source dataset is empty: {source}")
    return rows


def select_balanced_records(records: Iterable[SceneRecord], count: int) -> list[SceneRecord]:
    if count <= 0 or count % len(QUESTION_TYPES) != 0:
        raise ValueError(f"samples_per_seed must be a positive multiple of {len(QUESTION_TYPES)}")
    per_type = count // len(QUESTION_TYPES)
    groups = {name: [] for name in QUESTION_TYPES}
    for record in records:
        question_type = str(record.metadata.get("question_type"))
        if question_type in groups and len(groups[question_type]) < per_type:
            groups[question_type].append(record)
    missing = {name: per_type - len(rows) for name, rows in groups.items() if len(rows) < per_type}
    if missing:
        raise ValueError(f"source dataset cannot supply balanced pivot sample: {missing}")
    selected = [record for name in QUESTION_TYPES for record in groups[name]]
    return sorted(selected, key=lambda record: record.scene_id)


def build_seed_trials(
    model: BaseVLM,
    records: list[SceneRecord],
    *,
    seed: int,
    token_match_tolerance: int = 1,
) -> list[PivotTrial]:
    trials: list[PivotTrial] = []
    for index, record in enumerate(records):
        trials.extend(_a1_trials(model, record, seed=seed, index=index))
        trials.extend(
            _a2_trials(
                model,
                record,
                seed=seed,
                token_match_tolerance=token_match_tolerance,
            )
        )
        trials.extend(
            _a3_trials(
                model,
                record,
                seed=seed,
                token_match_tolerance=token_match_tolerance,
            )
        )
    return trials


def _a1_trials(
    model: BaseVLM, record: SceneRecord, *, seed: int, index: int
) -> list[PivotTrial]:
    edge = normalize_relation(dict(record.relations[0]))
    relation_nodes = {str(node["id"]): node for node in record.objects}
    target_shape = _unique_target(record.objects, ("category", "color", "size", "texture"), index)
    target_color = _unique_target(record.objects, ("category", "shape", "size", "texture"), index)
    specs = (
        (
            "object_shape",
            (
                "What shape does the {size} {texture} {color} object marked as {category} have?"
            ).format(**target_shape),
            str(target_shape["shape"]),
            SHAPES,
        ),
        (
            "attribute_color",
            (
                "What color is the {size} {texture} {shape} object marked as {category}?"
            ).format(**target_color),
            str(target_color["color"]),
            COLORS,
        ),
        (
            "direct_relation",
            (
                f"Where is {object_description(relation_nodes[edge['subject']])} relative to "
                f"{object_description(relation_nodes[edge['object']])}?"
            ),
            str(edge["relation"]),
            RELATIONS,
        ),
    )
    trials = []
    for task_index, (task, question, answer, vocabulary) in enumerate(specs):
        choices = _balanced_choices(
            answer,
            tuple(vocabulary),
            correct_index=(index + task_index) % 4,
        )
        if answer.casefold() in question.casefold().split():
            raise ValueError(f"{record.scene_id}:{task} question leaks the answer token")
        task_record = replace(
            record,
            scene_id=f"{record.scene_id}:{task}",
            question=question,
            answer=answer,
            choices=choices,
            metadata={
                **record.metadata,
                "pivot_experiment": "A1",
                "pivot_task": task,
                "source_scene_id": record.scene_id,
                "chance_accuracy": 0.25,
            },
        )
        evidence = "Evidence: image only."
        prepared = _prepared_input(
            model,
            task_record,
            condition="image_only",
            evidence=evidence,
            semantic_units=0,
            relation_count=0,
            serialization="image",
            fact_hash=None,
        )
        trials.append(
            PivotTrial(
                experiment="A1",
                seed=seed,
                source_scene_id=record.scene_id,
                trial_id=f"{record.scene_id}:{task}",
                condition="image_only",
                task=task,
                record=task_record,
                prepared=prepared,
                metadata={"correct_choice_index": choices.index(answer), "choices": list(choices)},
            )
        )
    return trials


def _a2_trials(
    model: BaseVLM,
    record: SceneRecord,
    *,
    seed: int,
    token_match_tolerance: int,
) -> list[PivotTrial]:
    object_facts = _object_facts(record)
    relation_facts, aliases = _aliased_relation_facts(record)
    full_facts = tuple(canonical_facts(record.objects, record.relations))
    if set(object_facts) | set(_canonical_relation_facts(record)) != set(full_facts):
        raise ValueError(
            f"{record.scene_id}: A2 object/relation evidence does not cover full facts"
        )
    object_text = "Objects: " + "; ".join(record.object_list) + "."
    relation_text = _serialize_relations(relation_facts, "natural_language")
    views = {
        "image_only": "Evidence: image only.",
        "object_evidence": f"Evidence: object facts only. {object_text}",
        "relation_evidence": f"Evidence: bound relation facts only. {relation_text}",
        "complete_evidence": (
            f"Evidence: complete object and relation facts. {object_text} {relation_text}"
        ),
    }
    matched = _match_prompt_budgets(
        model,
        record,
        views,
        tolerance=token_match_tolerance,
    )
    hashes = {
        "image_only": None,
        "object_evidence": facts_sha256(object_facts),
        "relation_evidence": facts_sha256(relation_facts),
        "complete_evidence": facts_sha256(full_facts),
    }
    semantic_units = {
        "image_only": 0,
        "object_evidence": len(object_facts),
        "relation_evidence": len(relation_facts),
        "complete_evidence": len(full_facts),
    }
    trials = []
    for condition in A2_CONDITIONS:
        prepared = _prepared_input(
            model,
            record,
            condition=condition,
            evidence=matched[condition]["text"],
            semantic_units=semantic_units[condition],
            relation_count=(
                len(relation_facts)
                if condition in {"relation_evidence", "complete_evidence"}
                else 0
            ),
            serialization="factorized_natural_language",
            fact_hash=hashes[condition],
            natural_tokens=matched[condition]["natural_tokens"],
            token_delta=matched[condition]["token_delta"],
            padding_units=matched[condition]["padding_units"],
        )
        trials.append(
            PivotTrial(
                experiment="A2",
                seed=seed,
                source_scene_id=record.scene_id,
                trial_id=record.scene_id,
                condition=condition,
                task=str(record.metadata["question_type"]),
                record=record,
                prepared=prepared,
                metadata={
                    "object_facts_sha256": facts_sha256(object_facts),
                    "relation_facts_sha256": facts_sha256(relation_facts),
                    "complete_facts_sha256": facts_sha256(full_facts),
                    "aliases": aliases,
                    "token_matched": True,
                },
            )
        )
    return trials


def _a3_trials(
    model: BaseVLM,
    record: SceneRecord,
    *,
    seed: int,
    token_match_tolerance: int,
) -> list[PivotTrial]:
    relation_facts, aliases = _aliased_relation_facts(record)
    digest = facts_sha256(relation_facts)
    views = {
        condition: f"Evidence: bound relation facts ({condition}). "
        + _serialize_relations(relation_facts, condition)
        for condition in A3_CONDITIONS
    }
    matched = _match_prompt_budgets(
        model,
        record,
        views,
        tolerance=token_match_tolerance,
    )
    trials = []
    for condition in A3_CONDITIONS:
        prepared = _prepared_input(
            model,
            record,
            condition=condition,
            evidence=matched[condition]["text"],
            semantic_units=len(relation_facts),
            relation_count=len(relation_facts),
            serialization=condition,
            fact_hash=digest,
            natural_tokens=matched[condition]["natural_tokens"],
            token_delta=matched[condition]["token_delta"],
            padding_units=matched[condition]["padding_units"],
        )
        trials.append(
            PivotTrial(
                experiment="A3",
                seed=seed,
                source_scene_id=record.scene_id,
                trial_id=record.scene_id,
                condition=condition,
                task=str(record.metadata["question_type"]),
                record=record,
                prepared=prepared,
                metadata={
                    "relation_facts_sha256": digest,
                    "aliases": aliases,
                    "token_matched": True,
                },
            )
        )
    return trials


def validate_seed_trials(trials: list[PivotTrial], *, samples_per_seed: int) -> dict[str, Any]:
    expected = {
        "A1": samples_per_seed * len(A1_TASKS),
        "A2": samples_per_seed * len(A2_CONDITIONS),
        "A3": samples_per_seed * len(A3_CONDITIONS),
    }
    counts = {name: sum(trial.experiment == name for trial in trials) for name in expected}
    assertions: dict[str, Any] = {
        "expected_trial_counts": counts == expected,
        "unique_trial_keys": len({trial.key for trial in trials}) == len(trials),
        "all_images_exist": all(Path(str(trial.prepared.image)).is_file() for trial in trials),
        "four_choice_a1": all(
            len(trial.record.choices) == 4 and trial.record.choices.count(trial.record.answer) == 1
            for trial in trials
            if trial.experiment == "A1"
        ),
        "a2_token_matched": _token_matched_by_scene(trials, "A2"),
        "a3_token_matched": _token_matched_by_scene(trials, "A3"),
        "a3_fact_identical": _one_value_by_scene(
            trials, "A3", lambda trial: trial.prepared.semantic_facts_sha256
        ),
        "training_disabled": True,
    }
    positions: dict[str, dict[int, int]] = {}
    for task in A1_TASKS:
        relevant = [trial for trial in trials if trial.experiment == "A1" and trial.task == task]
        positions[task] = {
            index: sum(trial.metadata["correct_choice_index"] == index for trial in relevant)
            for index in range(4)
        }
    assertions["a1_answer_positions_balanced"] = all(
        max(rows.values()) - min(rows.values()) <= 1 for rows in positions.values()
    )
    return {
        "passed": all(bool(value) for value in assertions.values()),
        "assertions": assertions,
        "counts": counts,
        "expected": expected,
        "answer_positions": positions,
    }


def _unique_target(
    objects: tuple[dict[str, Any], ...], fields: tuple[str, ...], preferred: int
) -> dict[str, Any]:
    for offset in range(len(objects)):
        candidate = objects[(preferred + offset) % len(objects)]
        signature = tuple(str(candidate[field]) for field in fields)
        if sum(tuple(str(node[field]) for field in fields) == signature for node in objects) == 1:
            return dict(candidate)
    raise ValueError(f"no unique primitive target for fields {fields}")


def _balanced_choices(
    answer: str, vocabulary: tuple[str, ...], *, correct_index: int
) -> tuple[str, ...]:
    distractors = [value for value in vocabulary if value != answer]
    start = int(hashlib.sha256(answer.encode("utf-8")).hexdigest()[:8], 16) % len(distractors)
    rotated = distractors[start:] + distractors[:start]
    choices = rotated[:3]
    choices.insert(correct_index, answer)
    return tuple(choices)


def _object_facts(record: SceneRecord) -> tuple[str, ...]:
    return tuple(
        "object|{id}|category={category}|shape={shape}|color={color}|size={size}|texture={texture}".format(
            **node
        )
        for node in sorted(record.objects, key=lambda item: str(item["id"]))
    )


def _canonical_relation_facts(record: SceneRecord) -> tuple[str, ...]:
    return tuple(
        f"relation|{edge['subject']}|{edge['relation']}|{edge['object']}"
        for raw in record.relations
        for edge in (normalize_relation(dict(raw)),)
    )


def _aliased_relation_facts(record: SceneRecord) -> tuple[tuple[str, ...], dict[str, str]]:
    query = dict(record.metadata.get("query", {}))
    subject = str(query.get("subject", ""))
    target = str(query.get("object", ""))
    answer_field = str(query.get("answer_field", ""))
    aliases: dict[str, str] = {}
    aliases[subject] = "question_subject" if answer_field == "relation" else "answer_object"
    aliases[target] = "question_reference_object"
    remaining = sorted(
        str(node["id"]) for node in record.objects if str(node["id"]) not in aliases
    )
    aliases.update(
        {
            identifier: f"intermediate_object_{index + 1}"
            for index, identifier in enumerate(remaining)
        }
    )
    facts = tuple(
        f"relation|{aliases[edge['subject']]}|{edge['relation']}|{aliases[edge['object']]}"
        for raw in record.relations
        for edge in (normalize_relation(dict(raw)),)
    )
    answer_leaked = any(record.answer.casefold() in fact.casefold() for fact in facts)
    if answer_field != "relation" and answer_leaked:
        raise ValueError(f"{record.scene_id}: relation evidence leaks a primitive answer")
    return facts, aliases


def _serialize_relations(facts: tuple[str, ...], serialization: str) -> str:
    triples = [fact.split("|")[1:] for fact in facts]
    if serialization == "natural_language":
        return "Relations: " + "; ".join(
            f"{subject.replace('_', ' ')} {_RELATION_PHRASES[relation]} {target.replace('_', ' ')}"
            for subject, relation, target in triples
        ) + "."
    if serialization == "triples":
        return "Relations: " + " ".join(
            f"({subject},{relation},{target})" for subject, relation, target in triples
        )
    if serialization == "json":
        payload = {
            "relations": [
                {"subject": subject, "relation": relation, "object": target}
                for subject, relation, target in triples
            ]
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    raise ValueError(f"unknown relation serialization: {serialization}")


def _match_prompt_budgets(
    model: BaseVLM,
    record: SceneRecord,
    views: dict[str, str],
    *,
    tolerance: int,
) -> dict[str, dict[str, Any]]:
    if tolerance < 0:
        raise ValueError("token_match_tolerance must be non-negative")
    texts = dict(views)
    natural = {name: _prompt_tokens(model, record, text) for name, text in texts.items()}
    current = dict(natural)
    padding = {name: 0 for name in texts}
    maximum_steps = max(128, (max(current.values()) - min(current.values()) + 1) * 16)
    for _ in range(maximum_steps):
        if max(current.values()) - min(current.values()) <= tolerance:
            break
        shortest = min(current, key=lambda name: (current[name], name))
        texts[shortest] += _PADDING_UNIT
        padding[shortest] += 1
        updated = _prompt_tokens(model, record, texts[shortest])
        if updated <= current[shortest]:
            raise ValueError("model token counter did not advance after pivot padding unit")
        current[shortest] = updated
    delta = max(current.values()) - min(current.values())
    if delta > tolerance:
        raise ValueError(
            f"could not token-match pivot views within tolerance {tolerance}: {current}"
        )
    return {
        name: {
            "text": texts[name],
            "natural_tokens": natural[name],
            "matched_tokens": current[name],
            "token_delta": delta,
            "padding_units": padding[name],
        }
        for name in views
    }


def _prompt_tokens(model: BaseVLM, record: SceneRecord, evidence: str) -> int:
    prompt = model.prompt_protocol.render(evidence, record.question, record.choices)
    return model.count_tokens(prompt)


def _prepared_input(
    model: BaseVLM,
    record: SceneRecord,
    *,
    condition: str,
    evidence: str,
    semantic_units: int,
    relation_count: int,
    serialization: str,
    fact_hash: str | None,
    natural_tokens: int | None = None,
    token_delta: int = 0,
    padding_units: int = 0,
) -> PreparedInput:
    prompt = model.prompt_protocol.render(evidence, record.question, record.choices)
    input_tokens = model.count_tokens(prompt)
    evidence_tokens = model.count_tokens(evidence)
    return PreparedInput(
        prompt=prompt,
        image=record.image,
        condition=condition,
        setting="token_matched" if padding_units else "natural",
        input_tokens=input_tokens,
        evidence_tokens=evidence_tokens,
        semantic_units=semantic_units,
        relation_count=relation_count,
        semantic_facts_sha256=fact_hash,
        serialization=serialization,
        padding_units=padding_units,
        token_match_delta=token_delta,
        prompt_protocol_id=model.prompt_protocol.protocol_id,
        prompt_protocol_sha256=model.prompt_protocol.sha256,
    )


def _token_matched_by_scene(trials: list[PivotTrial], experiment: str) -> bool:
    grouped: dict[tuple[int, str], list[int]] = {}
    for trial in trials:
        if trial.experiment == experiment:
            grouped.setdefault((trial.seed, trial.source_scene_id), []).append(
                trial.prepared.input_tokens
            )
    return bool(grouped) and all(max(values) - min(values) <= 1 for values in grouped.values())


def _one_value_by_scene(
    trials: list[PivotTrial],
    experiment: str,
    getter: Callable[[PivotTrial], Any],
) -> bool:
    grouped: dict[tuple[int, str], set[Any]] = {}
    for trial in trials:
        if trial.experiment == experiment:
            grouped.setdefault((trial.seed, trial.source_scene_id), set()).add(getter(trial))
    return bool(grouped) and all(len(values) == 1 for values in grouped.values())


__all__ = [
    "A1_TASKS",
    "A2_CONDITIONS",
    "A3_CONDITIONS",
    "PivotTrial",
    "build_seed_trials",
    "load_source_records",
    "select_balanced_records",
    "validate_seed_trials",
]
