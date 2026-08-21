"""Frozen 2x2x2 trial construction for PIVOT_EXP_A2."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from datasets.records import SceneRecord
from recoalign.models.vlm.base import BaseVLM, PreparedInput
from recoalign.synthetic_world.ontology import COLORS, SHAPES

from .relation_intervention import (
    RelationIntervention,
    build_relation_interventions,
    relation_by_condition,
)
from .semantic_scaffold import (
    EntityFact,
    SemanticScaffold,
    build_semantic_scaffolds,
    scaffold_by_condition,
)
from .serializers import canonical_inventory, inventory_sha256, serialize_facts

SEMANTIC_CONDITIONS = ("oracle_semantics", "corrupted_semantics")
RELATION_CONDITIONS = ("correct_relation", "corrupted_relation")
PRIMARY_SERIALIZATIONS = ("canonical_json", "canonical_triples")
REFERENCE_CONDITIONS = ("image_only", "natural_language")
MANIPULATION_TASKS = ("object_identity", "shape", "color", "entity_attribute_binding")
QUESTION_TYPES = ("multi_hop", "relation_reasoning", "object_reasoning", "attribute_reasoning")
PRIMARY_QUESTION_TYPES = ("multi_hop", "object_reasoning", "attribute_reasoning")
_PADDING_UNIT = " ."


@dataclass(frozen=True)
class CausalTrial:
    seed: int
    source_scene_id: str
    trial_id: str
    family: str
    condition: str
    task: str
    record: SceneRecord
    prepared: PreparedInput
    factors: dict[str, str | None]
    metadata: dict[str, Any]

    @property
    def key(self) -> str:
        return f"PIVOT_EXP_A2:{self.seed}:{self.family}:{self.trial_id}:{self.condition}"


def load_source_records(path: str | Path) -> list[SceneRecord]:
    source = Path(path)
    rows = [
        SceneRecord.from_dict(json.loads(line))
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"causal-separation source dataset is empty: {source}")
    return rows


def select_primary_records(records: list[SceneRecord]) -> list[SceneRecord]:
    selected = [
        record
        for record in records
        if str(record.metadata.get("question_type")) in PRIMARY_QUESTION_TYPES
    ]
    counts = Counter(str(record.metadata.get("question_type")) for record in selected)
    if counts != Counter({question_type: 30 for question_type in PRIMARY_QUESTION_TYPES}):
        raise ValueError(f"primary scene inventory differs from preregistration: {counts}")
    hop_counts = Counter(int(record.metadata.get("hop_depth", 0)) for record in selected)
    if hop_counts != Counter({1: 60, 2: 10, 3: 10, 4: 10}):
        raise ValueError(f"primary hop-depth inventory differs from preregistration: {hop_counts}")
    return sorted(selected, key=lambda record: record.scene_id)


def select_manipulation_records(records: list[SceneRecord]) -> list[SceneRecord]:
    selected: list[SceneRecord] = []
    for question_type in QUESTION_TYPES:
        group = sorted(
            (
                record
                for record in records
                if str(record.metadata.get("question_type")) == question_type
            ),
            key=lambda record: record.scene_id,
        )
        if len(group) < 5:
            raise ValueError(f"insufficient manipulation scenes for {question_type}")
        selected.extend(group[:5])
    return sorted(selected, key=lambda record: record.scene_id)


def build_seed_trials(
    model: BaseVLM,
    records: list[SceneRecord],
    *,
    seed: int,
    alias_salt: str,
    corruption_salt: str,
    token_match_tolerance: int = 1,
) -> list[CausalTrial]:
    primary = select_primary_records(records)
    manipulation = select_manipulation_records(records)
    trials: list[CausalTrial] = []
    for record in primary:
        scaffolds = build_semantic_scaffolds(
            record,
            seed=seed,
            alias_salt=alias_salt,
            corruption_salt=corruption_salt,
        )
        relations = build_relation_interventions(
            record,
            alias_by_object_id=scaffolds[0].alias_by_object_id,
            seed=seed,
            corruption_salt=corruption_salt,
        )
        trials.extend(
            _main_trials(
                model,
                record,
                seed=seed,
                scaffolds=scaffolds,
                relations=relations,
                token_match_tolerance=token_match_tolerance,
            )
        )
        trials.extend(
            _reference_trials(
                model,
                record,
                seed=seed,
                oracle=scaffolds[0],
                correct=relations[0],
            )
        )
    for record in manipulation:
        scaffolds = build_semantic_scaffolds(
            record,
            seed=seed,
            alias_salt=alias_salt,
            corruption_salt=corruption_salt,
        )
        trials.extend(
            _manipulation_trials(
                model,
                record,
                seed=seed,
                scaffolds=scaffolds,
                token_match_tolerance=token_match_tolerance,
            )
        )
    return trials


def _main_trials(
    model: BaseVLM,
    record: SceneRecord,
    *,
    seed: int,
    scaffolds: tuple[SemanticScaffold, SemanticScaffold],
    relations: tuple[RelationIntervention, RelationIntervention],
    token_match_tolerance: int,
) -> list[CausalTrial]:
    views: dict[str, dict[str, Any]] = {}
    for semantic_condition in SEMANTIC_CONDITIONS:
        semantic = scaffold_by_condition(scaffolds, semantic_condition)
        for relation_condition in RELATION_CONDITIONS:
            relation = relation_by_condition(relations, relation_condition)
            inventory = canonical_inventory(semantic.facts, relation.facts)
            for serialization in PRIMARY_SERIALIZATIONS:
                condition = _main_condition(
                    semantic_condition, relation_condition, serialization
                )
                views[condition] = {
                    "text": "Evidence: randomized semantic and relation intervention. "
                    + serialize_facts(semantic.facts, relation.facts, serialization),
                    "semantic": semantic,
                    "relation": relation,
                    "serialization": serialization,
                    "inventory": inventory,
                }
    matched = _match_prompt_budgets(
        model,
        record,
        {condition: payload["text"] for condition, payload in views.items()},
        tolerance=token_match_tolerance,
    )
    trials: list[CausalTrial] = []
    for condition, payload in views.items():
        semantic = payload["semantic"]
        relation = payload["relation"]
        serialization = str(payload["serialization"])
        inventory = payload["inventory"]
        prepared = _prepared_input(
            model,
            record,
            condition=condition,
            evidence=matched[condition]["text"],
            semantic_units=len(semantic.facts),
            relation_count=len(relation.facts),
            serialization=serialization,
            fact_hash=inventory_sha256(inventory),
            token_delta=matched[condition]["token_delta"],
            padding_units=matched[condition]["padding_units"],
        )
        trials.append(
            CausalTrial(
                seed=seed,
                source_scene_id=record.scene_id,
                trial_id=record.scene_id,
                family="main",
                condition=condition,
                task=str(record.metadata["question_type"]),
                record=record,
                prepared=prepared,
                factors={
                    "semantic": semantic.condition,
                    "relation": relation.condition,
                    "serialization": serialization,
                },
                metadata={
                    "canonical_inventory": list(inventory),
                    "inventory_sha256": inventory_sha256(inventory),
                    "oracle_semantic_facts": list(scaffolds[0].canonical_facts),
                    "active_semantic_facts": list(semantic.canonical_facts),
                    "correct_relation_facts": list(relations[0].canonical_facts),
                    "active_relation_facts": list(relation.canonical_facts),
                    "alias_by_object_id": dict(sorted(semantic.alias_by_object_id.items())),
                    "semantic_corruption_false": semantic.condition == "oracle_semantics"
                    or _semantic_corruption_false(scaffolds[0], semantic),
                    "relation_corruption_false": relation.condition == "correct_relation"
                    or set(relation.facts).isdisjoint(set(relations[0].facts)),
                    "relation_corruption_methods": list(relation.corruption_methods),
                    "explicit_answer_field_present": False,
                    "query_role_alias_present": False,
                    "natural_input_tokens": matched[condition]["natural_tokens"],
                    "matched_input_tokens": matched[condition]["matched_tokens"],
                },
            )
        )
    return trials


def _reference_trials(
    model: BaseVLM,
    record: SceneRecord,
    *,
    seed: int,
    oracle: SemanticScaffold,
    correct: RelationIntervention,
) -> list[CausalTrial]:
    views = {
        "image_only": ("Evidence: image only.", (), "image"),
        "natural_language": (
            "Evidence: secondary natural-language reference. "
            + serialize_facts(oracle.facts, correct.facts, "natural_language"),
            canonical_inventory(oracle.facts, correct.facts),
            "natural_language",
        ),
    }
    trials = []
    for condition, (evidence, inventory, serialization) in views.items():
        digest = inventory_sha256(inventory) if inventory else None
        prepared = _prepared_input(
            model,
            record,
            condition=condition,
            evidence=evidence,
            semantic_units=len(oracle.facts) if inventory else 0,
            relation_count=len(correct.facts) if inventory else 0,
            serialization=serialization,
            fact_hash=digest,
        )
        trials.append(
            CausalTrial(
                seed=seed,
                source_scene_id=record.scene_id,
                trial_id=record.scene_id,
                family="reference",
                condition=condition,
                task=str(record.metadata["question_type"]),
                record=record,
                prepared=prepared,
                factors={"semantic": None, "relation": None, "serialization": serialization},
                metadata={"primary_analysis_eligible": False},
            )
        )
    return trials


def _manipulation_trials(
    model: BaseVLM,
    record: SceneRecord,
    *,
    seed: int,
    scaffolds: tuple[SemanticScaffold, SemanticScaffold],
    token_match_tolerance: int,
) -> list[CausalTrial]:
    oracle = scaffolds[0]
    target = _unique_manipulation_target(oracle.facts)
    specs = _manipulation_specs(target, oracle.facts, seed=seed, scene_id=record.scene_id)
    trials: list[CausalTrial] = []
    for task, question, answer, choices in specs:
        task_record = replace(
            record,
            scene_id=f"{record.scene_id}:manipulation:{task}",
            question=question,
            answer=answer,
            choices=choices,
            metadata={
                **record.metadata,
                "causal_separation_family": "manipulation_check",
                "manipulation_task": task,
                "source_scene_id": record.scene_id,
            },
        )
        views = {
            scaffold.condition: "Evidence: semantic scaffold manipulation check. "
            + serialize_facts(scaffold.facts, (), "canonical_json")
            for scaffold in scaffolds
        }
        matched = _match_prompt_budgets(
            model,
            task_record,
            views,
            tolerance=token_match_tolerance,
        )
        for scaffold in scaffolds:
            condition = scaffold.condition
            inventory = canonical_inventory(scaffold.facts, ())
            prepared = _prepared_input(
                model,
                task_record,
                condition=condition,
                evidence=matched[condition]["text"],
                semantic_units=len(scaffold.facts),
                relation_count=0,
                serialization="canonical_json",
                fact_hash=inventory_sha256(inventory),
                token_delta=matched[condition]["token_delta"],
                padding_units=matched[condition]["padding_units"],
            )
            trials.append(
                CausalTrial(
                    seed=seed,
                    source_scene_id=record.scene_id,
                    trial_id=f"{record.scene_id}:{task}",
                    family="manipulation_check",
                    condition=condition,
                    task=task,
                    record=task_record,
                    prepared=prepared,
                    factors={
                        "semantic": condition,
                        "relation": None,
                        "serialization": "canonical_json",
                    },
                    metadata={
                        "primary_analysis_eligible": False,
                        "target_alias": target.alias,
                        "relation_fact_count": 0,
                        "semantic_corruption_false": condition == "oracle_semantics"
                        or _semantic_corruption_false(oracle, scaffold),
                        "natural_input_tokens": matched[condition]["natural_tokens"],
                        "matched_input_tokens": matched[condition]["matched_tokens"],
                    },
                )
            )
    return trials


def _manipulation_specs(
    target: EntityFact,
    facts: tuple[EntityFact, ...],
    *,
    seed: int,
    scene_id: str,
) -> tuple[tuple[str, str, str, tuple[str, ...]], ...]:
    aliases = tuple(fact.alias for fact in sorted(facts))
    description = f"{target.color} {target.shape}"
    descriptions = tuple(f"{fact.color} {fact.shape}" for fact in sorted(facts))
    tasks = (
        ("object_identity", target.alias, aliases, "Entity"),
        ("shape", target.shape, tuple(str(value) for value in SHAPES), "shape"),
        ("color", target.color, tuple(str(value) for value in COLORS), "color"),
        ("entity_attribute_binding", description, descriptions, "description"),
    )
    choices = {
        task: _four_choices(
            answer,
            candidates,
            prefix=prefix,
            correct_index=_choice_index(seed, scene_id, task),
        )
        for task, answer, candidates, prefix in tasks
    }
    return (
        (
            "object_identity",
            f"According to the evidence, which stable entity is the {description} object?",
            target.alias,
            choices["object_identity"],
        ),
        (
            "shape",
            f"According to the evidence, what shape belongs to {target.alias}?",
            target.shape,
            choices["shape"],
        ),
        (
            "color",
            f"According to the evidence, what color belongs to {target.alias}?",
            target.color,
            choices["color"],
        ),
        (
            "entity_attribute_binding",
            f"According to the evidence, which description belongs to {target.alias}?",
            description,
            choices["entity_attribute_binding"],
        ),
    )


def _unique_manipulation_target(facts: tuple[EntityFact, ...]) -> EntityFact:
    counts = Counter((fact.shape, fact.color) for fact in facts)
    for fact in sorted(facts):
        if counts[(fact.shape, fact.color)] == 1:
            return fact
    raise ValueError("manipulation check has no unique shape-color entity")


def _four_choices(
    answer: str,
    candidates: tuple[str, ...],
    *,
    prefix: str = "choice",
    correct_index: int,
) -> tuple[str, ...]:
    unique = [value for value in candidates if value != answer]
    index = 1
    while len(unique) < 3:
        filler = f"{prefix} {index}"
        if filler != answer and filler not in unique:
            unique.append(filler)
        index += 1
    choices = unique[:3]
    choices.insert(correct_index, answer)
    return tuple(choices)


def _choice_index(seed: int, scene_id: str, task: str) -> int:
    digest = hashlib.sha256(f"{seed}|{scene_id}|{task}|choice-position-v1".encode()).digest()
    return int.from_bytes(digest[:4], "big") % 4


def validate_seed_trials(
    trials: list[CausalTrial], *, expected_per_seed: int = 1060
) -> dict[str, Any]:
    main = [trial for trial in trials if trial.family == "main"]
    reference = [trial for trial in trials if trial.family == "reference"]
    manipulation = [trial for trial in trials if trial.family == "manipulation_check"]
    main_by_scene: dict[str, list[CausalTrial]] = defaultdict(list)
    for trial in main:
        main_by_scene[trial.source_scene_id].append(trial)
    manipulation_by_trial: dict[str, list[CausalTrial]] = defaultdict(list)
    for trial in manipulation:
        manipulation_by_trial[trial.trial_id].append(trial)
    assertions = {
        "prediction_design_count": len(trials) == expected_per_seed,
        "main_count": len(main) == 720,
        "reference_count": len(reference) == 180,
        "manipulation_count": len(manipulation) == 160,
        "unique_trial_keys": len({trial.key for trial in trials}) == len(trials),
        "all_images_exist": all(Path(str(trial.prepared.image)).is_file() for trial in trials),
        "complete_eight_cell_matrix": len(main_by_scene) == 90
        and all(len(rows) == 8 for rows in main_by_scene.values()),
        "main_token_matched": _token_matched(main_by_scene, tolerance=1),
        "json_triples_fact_identical": _format_facts_identical(main_by_scene),
        "all_semantic_corruptions_false": all(
            bool(trial.metadata["semantic_corruption_false"]) for trial in main
        ),
        "all_relation_corruptions_false": all(
            bool(trial.metadata["relation_corruption_false"]) for trial in main
        ),
        "semantic_scaffold_has_no_relation": all(
            all(
                not fact.startswith("relation|")
                for fact in trial.metadata["active_semantic_facts"]
            )
            for trial in main
        ),
        "semantic_scaffold_has_no_query_role": all(
            not bool(trial.metadata["query_role_alias_present"]) for trial in main
        ),
        "semantic_scaffold_has_no_explicit_answer": all(
            not bool(trial.metadata["explicit_answer_field_present"]) for trial in main
        ),
        "manipulation_pairs_complete": len(manipulation_by_trial) == 80
        and all(len(rows) == 2 for rows in manipulation_by_trial.values()),
        "manipulation_token_matched": all(
            max(row.prepared.input_tokens for row in rows)
            - min(row.prepared.input_tokens for row in rows)
            <= 1
            for rows in manipulation_by_trial.values()
        ),
        "manipulation_contains_no_relation": all(
            trial.prepared.relation_count == 0 for trial in manipulation
        ),
        "generation_protocol_identical": len(
            {trial.prepared.prompt_protocol_sha256 for trial in trials}
        )
        == 1,
    }
    return {
        "passed": all(assertions.values()),
        "assertions": assertions,
        "counts": {
            "all": len(trials),
            "main": len(main),
            "reference": len(reference),
            "manipulation_check": len(manipulation),
        },
        "primary_hop_depth": dict(
            sorted(Counter(int(trial.record.metadata["hop_depth"]) for trial in main[::8]).items())
        ),
    }


def _main_condition(semantic: str, relation: str, serialization: str) -> str:
    return f"S={semantic}|R={relation}|F={serialization}"


def _semantic_corruption_false(
    oracle: SemanticScaffold, corrupted: SemanticScaffold
) -> bool:
    truth = {fact.alias: (fact.shape, fact.color) for fact in oracle.facts}
    return all(truth[fact.alias] != (fact.shape, fact.color) for fact in corrupted.facts)


def _match_prompt_budgets(
    model: BaseVLM,
    record: SceneRecord,
    views: dict[str, str],
    *,
    tolerance: int,
) -> dict[str, dict[str, Any]]:
    if tolerance < 0:
        raise ValueError("token tolerance must be non-negative")
    texts = dict(views)
    natural = {name: _prompt_tokens(model, record, text) for name, text in texts.items()}
    current = dict(natural)
    padding = {name: 0 for name in texts}
    maximum_steps = max(256, (max(current.values()) - min(current.values()) + 1) * 16)
    for _ in range(maximum_steps):
        if max(current.values()) - min(current.values()) <= tolerance:
            break
        shortest = min(current, key=lambda name: (current[name], name))
        texts[shortest] += _PADDING_UNIT
        padding[shortest] += 1
        updated = _prompt_tokens(model, record, texts[shortest])
        if updated <= current[shortest]:
            raise ValueError("tokenizer did not advance after punctuation padding")
        current[shortest] = updated
    delta = max(current.values()) - min(current.values())
    if delta > tolerance:
        raise ValueError(f"could not token-match causal cells: {current}")
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
    token_delta: int = 0,
    padding_units: int = 0,
) -> PreparedInput:
    prompt = model.prompt_protocol.render(evidence, record.question, record.choices)
    return PreparedInput(
        prompt=prompt,
        image=record.image,
        condition=condition,
        setting="token_matched" if padding_units else "natural",
        input_tokens=model.count_tokens(prompt),
        evidence_tokens=model.count_tokens(evidence),
        semantic_units=semantic_units,
        relation_count=relation_count,
        semantic_facts_sha256=fact_hash,
        serialization=serialization,
        padding_units=padding_units,
        token_match_delta=token_delta,
        prompt_protocol_id=model.prompt_protocol.protocol_id,
        prompt_protocol_sha256=model.prompt_protocol.sha256,
    )


def _token_matched(groups: dict[str, list[CausalTrial]], *, tolerance: int) -> bool:
    return bool(groups) and all(
        max(trial.prepared.input_tokens for trial in rows)
        - min(trial.prepared.input_tokens for trial in rows)
        <= tolerance
        for rows in groups.values()
    )


def _format_facts_identical(groups: dict[str, list[CausalTrial]]) -> bool:
    for rows in groups.values():
        by_factors: dict[tuple[str | None, str | None], set[str | None]] = defaultdict(set)
        for trial in rows:
            key = (trial.factors["semantic"], trial.factors["relation"])
            by_factors[key].add(trial.prepared.semantic_facts_sha256)
        if len(by_factors) != 4 or any(len(values) != 1 for values in by_factors.values()):
            return False
    return True


def design_digest(trials: list[CausalTrial]) -> str:
    payload = [
        {
            "key": trial.key,
            "condition": trial.condition,
            "factors": trial.factors,
            "prompt_sha256": hashlib.sha256(trial.prepared.prompt.encode("utf-8")).hexdigest(),
            "facts_sha256": trial.prepared.semantic_facts_sha256,
        }
        for trial in sorted(trials, key=lambda row: row.key)
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "CausalTrial",
    "MANIPULATION_TASKS",
    "PRIMARY_SERIALIZATIONS",
    "REFERENCE_CONDITIONS",
    "RELATION_CONDITIONS",
    "SEMANTIC_CONDITIONS",
    "build_seed_trials",
    "design_digest",
    "load_source_records",
    "select_manipulation_records",
    "select_primary_records",
    "validate_seed_trials",
]
