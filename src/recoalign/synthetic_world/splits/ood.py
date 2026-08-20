"""Controlled, leakage-audited splits for EXP003 compositional generalization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datasets.records import SceneRecord
from recoalign.synthetic_world.ontology import (
    CATEGORIES,
    COLORS,
    MULTIHOP_RELATIONS,
    SHAPES,
    SIZES,
    TEXTURES,
)
from recoalign.synthetic_world.questions import (
    canonical_facts,
    caption_from_world,
    facts_sha256,
    generate_question,
    object_description,
    object_list_from_world,
)
from recoalign.synthetic_world.scene_graph import SceneGraph

if TYPE_CHECKING:
    from recoalign.synthetic_world.generator.generator import SyntheticWorldGenerator

SPLIT_NAMES = ("iid", "composition_ood", "relation_ood", "hop_ood")
PARTITIONS = ("train", "test")
MINIMUM_PARTITION_SIZE = 8

TRAIN_RELATION_COMBINATIONS = (
    ("left", "above"),
    ("above", "left"),
    ("front", "left"),
    ("behind", "above"),
)
TEST_RELATION_COMBINATIONS = (
    ("left", "behind"),
    ("behind", "left"),
    ("above", "front"),
    ("front", "above"),
)
RELATION_COMBINATION_CHOICES = tuple(
    "+".join(values)
    for values in (*TRAIN_RELATION_COMBINATIONS, *TEST_RELATION_COMBINATIONS)
)


@dataclass(frozen=True)
class ControlledSplit:
    """One named train/test split plus its validated scientific assertions."""

    name: str
    train: tuple[SceneRecord, ...]
    test: tuple[SceneRecord, ...]
    validation: dict[str, Any]


@dataclass(frozen=True)
class OODSplitSuite:
    """The four EXP003 split families generated from a single deterministic seed."""

    seed: int
    splits: dict[str, ControlledSplit]
    validation: dict[str, Any]
    manifest: dict[str, Any] | None = None


def build_ood_split_suite(
    generator: SyntheticWorldGenerator,
    *,
    train_count: int,
    test_count: int,
    seed: int,
    output_dir: str | Path | None = None,
) -> OODSplitSuite:
    """Build controlled IID, composition, relation-combination, and hop splits.

    The allocation is constructive rather than random: every held-out property is declared before
    records are built, so no test result can influence split membership.
    """

    if min(train_count, test_count) < MINIMUM_PARTITION_SIZE:
        raise ValueError(
            "controlled OOD partitions require at least "
            f"{MINIMUM_PARTITION_SIZE} samples for primitive coverage"
        )
    if seed < 0:
        raise ValueError("split seed must be non-negative")

    raw: dict[str, dict[str, list[SceneRecord]]] = {
        name: {partition: [] for partition in PARTITIONS} for name in SPLIT_NAMES
    }
    for partition, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            raw["iid"][partition].append(
                _iid_record(generator, seed, partition, index)
            )
            raw["composition_ood"][partition].append(
                _composition_record(generator, seed, partition, index)
            )
            raw["relation_ood"][partition].append(
                _relation_record(generator, seed, partition, index)
            )
            raw["hop_ood"][partition].append(
                _hop_record(generator, seed, partition, index)
            )

    validation = validate_ood_split_suite(raw)
    if not validation["valid"]:
        raise ValueError(
            "controlled OOD split validation failed: "
            + "; ".join(validation["violations"])
        )

    materialized = raw
    manifest: dict[str, Any] | None = None
    if output_dir is not None:
        materialized, manifest = _materialize_suite(generator, raw, Path(output_dir), validation)
        post_validation = validate_ood_split_suite(
            materialized, check_images=generator.config.write_images
        )
        if not post_validation["valid"]:
            raise ValueError(
                "materialized OOD split validation failed: "
                + "; ".join(post_validation["violations"])
            )
        validation = post_validation

    splits = {
        name: ControlledSplit(
            name=name,
            train=tuple(materialized[name]["train"]),
            test=tuple(materialized[name]["test"]),
            validation=dict(validation["splits"][name]),
        )
        for name in SPLIT_NAMES
    }
    return OODSplitSuite(seed=seed, splits=splits, validation=validation, manifest=manifest)


def controlled_composition_signature(record: SceneRecord) -> str:
    """Return the semantic composition held out by EXP003.

    Size, texture, and category are nuisance factors.  The tested composition is the ordered
    color-shape binding at the query endpoints, the ordered relation program, and hop depth.
    """

    query = dict(record.metadata["query"])
    lookup = {str(node["id"]): node for node in record.objects}
    subject = lookup[str(query["subject"])]
    target = lookup[str(query["object"])]
    relations = _relation_sequence(record)
    return "|".join(
        (
            f"{subject['color']}:{subject['shape']}",
            "+".join(relations),
            f"{target['color']}:{target['shape']}",
            str(record.metadata["hop_depth"]),
            str(record.metadata["question_type"]),
        )
    )


def relation_combination_signature(record: SceneRecord) -> str:
    return "+".join(_relation_sequence(record))


def validate_ood_split_suite(
    splits: dict[str, dict[str, list[SceneRecord]]], *, check_images: bool = False
) -> dict[str, Any]:
    """Fail closed on overlap, missing primitives, malformed records, or policy drift."""

    from recoalign.synthetic_world.validation import validate_dataset

    violations: list[str] = []
    reports: dict[str, Any] = {}
    unknown = sorted(set(splits) - set(SPLIT_NAMES))
    missing = sorted(set(SPLIT_NAMES) - set(splits))
    if unknown:
        violations.append(f"unknown split families: {unknown}")
    if missing:
        violations.append(f"missing split families: {missing}")

    for name in SPLIT_NAMES:
        if name not in splits:
            continue
        train = list(splits[name].get("train", ()))
        test = list(splits[name].get("test", ()))
        local: list[str] = []
        if not train or not test:
            local.append("train and test partitions must both be non-empty")
        identifiers_train = {row.scene_id for row in train}
        identifiers_test = {row.scene_id for row in test}
        identifier_overlap = sorted(identifiers_train & identifiers_test)
        if identifier_overlap:
            local.append("sample IDs overlap across train and test")
        try:
            validate_dataset([*train, *test], check_images=check_images)
        except (TypeError, ValueError) as exc:
            local.append(f"dataset validation failed: {exc}")

        train_primitives = _primitive_inventory(train)
        test_primitives = _primitive_inventory(test)
        missing_primitives = {
            key: sorted(test_primitives[key] - train_primitives[key])
            for key in train_primitives
            if test_primitives[key] - train_primitives[key]
        }
        primitive_coverage = not missing_primitives
        if not primitive_coverage:
            local.append(f"test primitives absent from training: {missing_primitives}")

        train_compositions = {controlled_composition_signature(row) for row in train}
        test_compositions = {controlled_composition_signature(row) for row in test}
        composition_overlap = sorted(train_compositions & test_compositions)
        train_relation_combinations = {relation_combination_signature(row) for row in train}
        test_relation_combinations = {relation_combination_signature(row) for row in test}
        relation_overlap = sorted(train_relation_combinations & test_relation_combinations)
        train_depths = {int(row.metadata["hop_depth"]) for row in train}
        test_depths = {int(row.metadata["hop_depth"]) for row in test}
        depth_overlap = sorted(train_depths & test_depths)

        if name == "iid" and not test_compositions <= train_compositions:
            local.append("IID test contains a composition absent from IID training")
        if name == "composition_ood" and composition_overlap:
            local.append("composition OOD train/test compositions overlap")
        if name == "relation_ood":
            if relation_overlap:
                local.append("relation OOD train/test relation combinations overlap")
            if train_depths != {2} or test_depths != {2}:
                local.append("relation OOD must hold hop depth fixed at two")
        if name == "hop_ood" and (train_depths != {1, 2} or test_depths != {3, 4}):
            local.append("hop OOD must use train depths {1,2} and test depths {3,4}")

        train_worlds = {str(row.metadata["world_sha256"]) for row in train}
        test_worlds = {str(row.metadata["world_sha256"]) for row in test}
        reports[name] = {
            "valid": not local,
            "violations": local,
            "counts": {"train": len(train), "test": len(test)},
            "sample_id_overlap": identifier_overlap,
            "primitive_coverage": primitive_coverage,
            "missing_primitives": missing_primitives,
            "composition_overlap": composition_overlap,
            "relation_combination_overlap": relation_overlap,
            "hop_depth_overlap": depth_overlap,
            "train_hop_depths": sorted(train_depths),
            "test_hop_depths": sorted(test_depths),
            "world_hash_overlap": sorted(train_worlds & test_worlds),
            "train_composition_count": len(train_compositions),
            "test_composition_count": len(test_compositions),
            "train_relation_combinations": sorted(train_relation_combinations),
            "test_relation_combinations": sorted(test_relation_combinations),
        }
        violations.extend(f"{name}: {message}" for message in local)

    assertions = {
        "composition_overlap": bool(
            reports.get("composition_ood", {}).get("composition_overlap", True)
        ),
        "relation_combination_overlap": bool(
            reports.get("relation_ood", {}).get("relation_combination_overlap", True)
        ),
        "hop_depth_overlap": bool(reports.get("hop_ood", {}).get("hop_depth_overlap", True)),
        "primitive_coverage": all(
            bool(reports.get(name, {}).get("primitive_coverage")) for name in SPLIT_NAMES
        ),
        "sample_id_overlap": any(
            bool(reports.get(name, {}).get("sample_id_overlap")) for name in SPLIT_NAMES
        ),
    }
    return {
        "valid": not violations,
        "violations": violations,
        "splits": reports,
        "assertions": assertions,
    }


def _iid_record(
    generator: SyntheticWorldGenerator, seed: int, partition: str, index: int
) -> SceneRecord:
    depth = 1 + index % 4
    relation = MULTIHOP_RELATIONS[index % len(MULTIHOP_RELATIONS)]
    return _record(
        generator,
        seed=seed,
        split_name="iid",
        partition=partition,
        index=index,
        depth=depth,
        relation_sequence=(relation,) * depth,
        identity_offset=0,
        color_offset=0,
        nuisance_offset=0 if partition == "train" else 3,
        anti_memorization=None,
    )


def _composition_record(
    generator: SyntheticWorldGenerator, seed: int, partition: str, index: int
) -> SceneRecord:
    depth = 1 + index % 4
    relation = MULTIHOP_RELATIONS[index % len(MULTIHOP_RELATIONS)]
    anti = "object_identity_swap" if index % 2 == 0 else "attribute_transfer"
    identity_offset = 1 if partition == "test" and anti == "object_identity_swap" else 0
    color_offset = 2 if partition == "test" and anti == "attribute_transfer" else 0
    return _record(
        generator,
        seed=seed,
        split_name="composition_ood",
        partition=partition,
        index=index,
        depth=depth,
        relation_sequence=(relation,) * depth,
        identity_offset=identity_offset,
        color_offset=color_offset,
        nuisance_offset=0 if partition == "train" else 1,
        anti_memorization=anti,
    )


def _relation_record(
    generator: SyntheticWorldGenerator, seed: int, partition: str, index: int
) -> SceneRecord:
    combinations = (
        TRAIN_RELATION_COMBINATIONS if partition == "train" else TEST_RELATION_COMBINATIONS
    )
    return _record(
        generator,
        seed=seed,
        split_name="relation_ood",
        partition=partition,
        index=index,
        depth=2,
        relation_sequence=combinations[index % len(combinations)],
        identity_offset=0,
        color_offset=0,
        nuisance_offset=0 if partition == "train" else 2,
        anti_memorization="relation_recombination",
    )


def _hop_record(
    generator: SyntheticWorldGenerator, seed: int, partition: str, index: int
) -> SceneRecord:
    depth = (1 + index % 2) if partition == "train" else (3 + index % 2)
    relation = MULTIHOP_RELATIONS[index % len(MULTIHOP_RELATIONS)]
    return _record(
        generator,
        seed=seed,
        split_name="hop_ood",
        partition=partition,
        index=index,
        depth=depth,
        relation_sequence=(relation,) * depth,
        identity_offset=0,
        color_offset=0,
        nuisance_offset=0 if partition == "train" else 2,
        anti_memorization=None,
    )


def _record(
    generator: SyntheticWorldGenerator,
    *,
    seed: int,
    split_name: str,
    partition: str,
    index: int,
    depth: int,
    relation_sequence: tuple[str, ...],
    identity_offset: int,
    color_offset: int,
    nuisance_offset: int,
    anti_memorization: str | None,
) -> SceneRecord:
    if len(relation_sequence) != depth:
        raise ValueError("relation sequence length must equal hop depth")
    sample_seed = _derived_seed(seed, split_name, partition, index)
    objects = tuple(
        _object(index, position, identity_offset, color_offset, nuisance_offset)
        for position in range(depth + 1)
    )
    relations = tuple(
        {
            "subject": f"obj{position + 1}",
            "relation": relation,
            "object": f"obj{position + 2}",
        }
        for position, relation in enumerate(relation_sequence)
    )
    graph = SceneGraph(objects, relations)
    if len(set(relation_sequence)) == 1:
        question_type = "relation_reasoning" if depth == 1 else "multi_hop"
        question = generate_question(graph, question_type=question_type)
        question_text = question.question
        answer = question.answer
        choices = question.choices
        query = question.query
    else:
        question_type = "multi_hop"
        subject = str(objects[0]["id"])
        target = str(objects[-1]["id"])
        answer = "+".join(relation_sequence)
        question_text = (
            f"Following the {depth}-edge path from {object_description(objects[0])} to "
            f"{object_description(objects[-1])}, what is the ordered sequence of stated relations?"
        )
        choices = RELATION_COMBINATION_CHOICES
        query = {
            "subject": subject,
            "object": target,
            "answer_field": "relation_sequence",
            "supporting_edges": list(range(depth)),
            "relation_sequence": list(relation_sequence),
        }

    facts = canonical_facts(objects, relations)
    digest = facts_sha256(facts)
    world_payload = {"objects": objects, "relations": relations, "seed": sample_seed}
    world_sha = hashlib.sha256(
        json.dumps(world_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    pair_id = f"{split_name}:{index:06d}"
    metadata: dict[str, Any] = {
        "generator": generator.version,
        "seed": seed,
        "sample_seed": sample_seed,
        "split": partition,
        "ood_split": split_name,
        "partition": partition,
        "pair_id": pair_id,
        "question_type": question_type,
        "hop_depth": depth,
        "difficulty": depth,
        "primary_relation": relation_sequence[0],
        "relation_sequence": list(relation_sequence),
        "relation_combination": "+".join(relation_sequence),
        "query": query,
        "world_sha256": world_sha,
        "canonical_facts": list(facts),
        "canonical_facts_sha256": digest,
        "information_control": {
            "graph_facts_sha256": digest,
            "caption_facts_sha256": digest,
            "caption_graph_equivalent": True,
            "caption_generator": "deterministic-template-v2",
        },
        "render": {
            "renderer": generator.renderer.version,
            "resolution": generator.config.image_size,
            "style": generator.config.style,
        },
    }
    if anti_memorization is not None:
        metadata["anti_memorization"] = anti_memorization
    record = SceneRecord(
        scene_id=f"exp003_{seed}_{split_name}_{partition}_{index:06d}",
        image="",
        objects=objects,
        relations=relations,
        question=question_text,
        answer=answer,
        choices=choices,
        metadata=metadata,
        object_list=object_list_from_world(objects),
        attributes=tuple(
            {
                "object_id": str(node["id"]),
                **{
                    field: node[field]
                    for field in ("category", "shape", "color", "size", "texture")
                },
            }
            for node in objects
        ),
        caption=caption_from_world(objects, relations),
        split=partition,
    )
    metadata["composition"] = controlled_composition_signature(record)
    return record


def _object(
    template_index: int,
    position: int,
    identity_offset: int,
    color_offset: int,
    nuisance_offset: int,
) -> dict[str, str]:
    base = template_index + position
    return {
        "id": f"obj{position + 1}",
        "category": CATEGORIES[(base + nuisance_offset) % len(CATEGORIES)],
        "shape": SHAPES[(base + identity_offset) % len(SHAPES)],
        "color": COLORS[(base + color_offset) % len(COLORS)],
        "size": SIZES[(base + 2 * nuisance_offset) % len(SIZES)],
        "texture": TEXTURES[(2 * base + nuisance_offset) % len(TEXTURES)],
    }


def _relation_sequence(record: SceneRecord) -> tuple[str, ...]:
    declared = record.metadata.get("relation_sequence")
    if isinstance(declared, list) and declared:
        return tuple(str(value) for value in declared)
    return tuple(str(edge["relation"]) for edge in record.relations)


def _primitive_inventory(records: list[SceneRecord]) -> dict[str, set[str]]:
    result = {
        "category": set(),
        "shape": set(),
        "color": set(),
        "size": set(),
        "texture": set(),
        "relation": set(),
    }
    for record in records:
        for node in record.objects:
            for field in ("category", "shape", "color", "size", "texture"):
                result[field].add(str(node[field]))
        result["relation"].update(_relation_sequence(record))
    return result


def _materialize_suite(
    generator: SyntheticWorldGenerator,
    raw: dict[str, dict[str, list[SceneRecord]]],
    root: Path,
    validation: dict[str, Any],
) -> tuple[dict[str, dict[str, list[SceneRecord]]], dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    manifest_root = root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    materialized: dict[str, dict[str, list[SceneRecord]]] = {
        name: {} for name in SPLIT_NAMES
    }
    split_manifests: dict[str, Any] = {}
    for name in SPLIT_NAMES:
        for partition in PARTITIONS:
            materialized[name][partition] = generator.materialize(
                raw[name][partition], root / name / partition
            )
        files = _file_entries(root, root / name)
        split_manifest = {
            "schema_version": 1,
            "experiment_id": "EXP003",
            "generator": generator.version,
            "seed": int(materialized[name]["train"][0].metadata["seed"]),
            "split": name,
            "counts": {
                partition: len(materialized[name][partition]) for partition in PARTITIONS
            },
            "policy": _policy_description(name),
            "validation": validation["splits"][name],
            "files": files,
        }
        _write_json(manifest_root / f"{name}.json", split_manifest)
        split_manifests[name] = {
            "path": f"manifests/{name}.json",
            "sha256": _sha256(manifest_root / f"{name}.json"),
            "counts": split_manifest["counts"],
        }
    manifest = {
        "schema_version": 1,
        "experiment_id": "EXP003",
        "generator": generator.version,
        "seed": int(next(iter(materialized.values()))["train"][0].metadata["seed"]),
        "deterministic_from_seed": True,
        "allocation": "constructive-predeclared-v1",
        "test_information_used_for_allocation": False,
        "splits": split_manifests,
        "validation": validation,
        "manifest_assertions": validation["assertions"],
    }
    _write_json(root / "manifest.json", manifest)
    manifest["sha256"] = _sha256(root / "manifest.json")
    return materialized, manifest


def _file_entries(root: Path, split_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(split_root.rglob("*"))
        if path.is_file()
    ]


def _policy_description(name: str) -> dict[str, Any]:
    policies = {
        "iid": {
            "train": "known color-shape/relation/depth compositions",
            "test": "same composition support with changed nuisance factors",
        },
        "composition_ood": {
            "train": "identity binding offset 0",
            "test": "identity binding offsets 1/2; all primitives retained",
        },
        "relation_ood": {
            "train": ["+".join(values) for values in TRAIN_RELATION_COMBINATIONS],
            "test": ["+".join(values) for values in TEST_RELATION_COMBINATIONS],
        },
        "hop_ood": {"train_hops": [1, 2], "test_hops": [3, 4]},
    }
    return policies[name]


def _derived_seed(seed: int, split_name: str, partition: str, index: int) -> int:
    digest = hashlib.sha256(
        f"EXP003:{seed}:{split_name}:{partition}:{index}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


__all__ = [
    "ControlledSplit",
    "MINIMUM_PARTITION_SIZE",
    "OODSplitSuite",
    "PARTITIONS",
    "SPLIT_NAMES",
    "build_ood_split_suite",
    "controlled_composition_signature",
    "relation_combination_signature",
    "validate_ood_split_suite",
]
