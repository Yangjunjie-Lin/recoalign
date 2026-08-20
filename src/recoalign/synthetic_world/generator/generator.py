"""World-first generation for the ReCoAlign compositional instrument."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jsonschema

from datasets.records import SceneRecord
from recoalign.synthetic_world.ontology import (
    CATEGORIES,
    COLORS,
    MULTIHOP_RELATIONS,
    RELATIONS,
    SHAPES,
    SIZES,
    TEXTURES,
    ObjectSpec,
    RelationSpec,
)
from recoalign.synthetic_world.questions import (
    QUESTION_TYPES,
    canonical_facts,
    caption_from_world,
    facts_sha256,
    generate_question,
    object_list_from_world,
)
from recoalign.synthetic_world.renderer import DeterministicRenderer, RendererConfig
from recoalign.synthetic_world.scene_graph import SceneGraph
from recoalign.synthetic_world.splits import apply_split, composition_signature


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 7
    image_size: int = 192
    write_images: bool = True
    style: str = "flat"
    min_hops: int = 1
    max_hops: int = 4

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("generator seed must be non-negative")
        if not 1 <= self.min_hops <= self.max_hops <= 4:
            raise ValueError("hop bounds must satisfy 1 <= min_hops <= max_hops <= 4")
        RendererConfig(resolution=self.image_size, style=self.style)


class SyntheticWorldGenerator:
    """Generate world state, then derive graph, image, caption, and QA from it."""

    version = "recoalign.synthetic_world.v2"

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self.config = config or GeneratorConfig()
        self.renderer = DeterministicRenderer(
            RendererConfig(resolution=self.config.image_size, style=self.config.style)
        )

    def generate(
        self,
        count: int,
        *,
        output_dir: str | Path | None = None,
        seed: int | None = None,
        split: str = "test",
        relation_order: Iterable[str] | None = None,
        split_strategy: str | None = None,
    ) -> list[SceneRecord]:
        if count <= 0:
            raise ValueError("count must be positive")
        base_seed = self.config.seed if seed is None else seed
        if base_seed < 0:
            raise ValueError("seed must be non-negative")
        forced_relations = tuple(relation_order or ())
        if any(value not in RELATIONS for value in forced_relations):
            raise ValueError(f"relation_order values must be in {RELATIONS}")
        records = [
            self._build_record(
                index,
                base_seed=base_seed,
                split=split,
                forced_relation=(
                    forced_relations[index % len(forced_relations)] if forced_relations else None
                ),
            )
            for index in range(count)
        ]
        if split_strategy is not None:
            records = apply_split(records, split_strategy)
        else:
            records = [
                replace(
                    record,
                    metadata={
                        **record.metadata,
                        "composition": composition_signature(record),
                    },
                )
                for record in records
            ]
        if output_dir is not None:
            records = self.materialize(records, output_dir)
        return records

    def generate_ood_splits(
        self,
        train_count: int,
        test_count: int,
        *,
        output_dir: str | Path | None = None,
        seed: int | None = None,
    ) -> tuple[list[SceneRecord], list[SceneRecord]]:
        """Compatibility API for EXP003 with explicitly disjoint composition signatures."""

        if train_count <= 0 or test_count <= 0:
            raise ValueError("OOD split counts must be positive")
        # Preserve the preregistered v1 capacity guard so old configurations fail identically.
        if test_count >= 360:
            raise ValueError("requested OOD compositions exceed the registered capacity of 359")
        base_seed = self.config.seed if seed is None else seed
        train = self.generate(train_count, seed=base_seed, split="train")
        used = {str(record.metadata["composition"]) for record in train}
        test: list[SceneRecord] = []
        test_used: set[str] = set()
        maximum_attempts = max(2000, test_count * 200)
        for offset in range(maximum_attempts):
            if len(test) >= test_count:
                break
            candidate = self._build_record(
                offset,
                base_seed=base_seed + 1000,
                split="ood_test",
                forced_relation=MULTIHOP_RELATIONS[offset % len(MULTIHOP_RELATIONS)],
            )
            candidate = replace(
                candidate,
                metadata={
                    **candidate.metadata,
                    "composition": composition_signature(candidate),
                },
            )
            signature = str(candidate.metadata["composition"])
            if signature in used or signature in test_used:
                continue
            test_used.add(signature)
            test.append(
                replace(
                    candidate,
                    scene_id=f"ood_scene_{len(test):05d}",
                    split="ood_test",
                    metadata={**candidate.metadata, "split": "ood_test"},
                )
            )
        if len(test) != test_count:
            raise RuntimeError(
                f"could not sample {test_count} unique OOD compositions in "
                f"{maximum_attempts} attempts"
            )
        if output_dir is not None:
            root = Path(output_dir)
            train = self.materialize(train, root / "train")
            test = self.materialize(test, root / "test")
            write_jsonl(root / "train.jsonl", train)
            write_jsonl(root / "test.jsonl", test)
            overlap = sorted(
                {str(row.metadata["composition"]) for row in train}
                & {str(row.metadata["composition"]) for row in test}
            )
            _write_json(
                root / "manifest.json",
                {
                    "generator": self.version,
                    "train_count": len(train),
                    "test_count": len(test),
                    "seed": base_seed,
                    "train_compositions": sorted(
                        str(record.metadata["composition"]) for record in train
                    ),
                    "test_compositions": sorted(
                        str(record.metadata["composition"]) for record in test
                    ),
                    "composition_overlap": bool(overlap),
                    "overlap_values": overlap,
                },
            )
        return train, test

    def materialize(
        self, records: list[SceneRecord], output_dir: str | Path
    ) -> list[SceneRecord]:
        """Write reconstructable per-sample bundles and aggregate indexes."""

        root = Path(output_dir)
        image_root = root / "images"
        sample_root = root / "samples"
        image_root.mkdir(parents=True, exist_ok=True)
        sample_root.mkdir(parents=True, exist_ok=True)
        materialized: list[SceneRecord] = []
        for record in records:
            bundle = sample_root / record.scene_id
            bundle.mkdir(parents=True, exist_ok=True)
            image = ""
            metadata = dict(record.metadata)
            if self.config.write_images:
                aggregate_image = image_root / f"{record.scene_id}.png"
                bundle_image = bundle / "image.png"
                self.renderer.render(aggregate_image, list(record.objects), list(record.relations))
                shutil.copyfile(aggregate_image, bundle_image)
                image = aggregate_image.as_posix()
                metadata["image_sha256"] = _sha256_file(aggregate_image)
            row = replace(record, image=image, metadata=metadata)
            graph_payload = {
                "objects": [dict(item) for item in row.objects],
                "relations": [dict(item) for item in row.to_dict()["relations"]],
            }
            _write_json(bundle / "scene.json", row.to_dict())
            _write_json(bundle / "graph.json", graph_payload)
            bundle_metadata = {
                "id": row.scene_id,
                **metadata,
                "files": {
                    "image": "image.png" if image else None,
                    "scene": "scene.json",
                    "graph": "graph.json",
                },
            }
            _write_json(bundle / "metadata.json", bundle_metadata)
            materialized.append(row)
        write_jsonl(root / "dataset.jsonl", materialized)
        write_jsonl(root / "metadata.jsonl", materialized)
        _write_json(root / "scenes.json", [row.to_dict() for row in materialized])
        dataset_sha = _sha256_file(root / "dataset.jsonl")
        manifest = self._manifest(materialized, dataset_sha)
        _write_json(root / "manifest.json", manifest)
        return materialized

    def rebuild_image(self, record: SceneRecord, output_path: str | Path) -> dict[str, Any]:
        """Re-render a sample and compare bytes with its recorded image checksum."""

        path = self.renderer.render(output_path, list(record.objects), list(record.relations))
        actual = _sha256_file(path)
        expected = record.metadata.get("image_sha256")
        return {
            "path": path.as_posix(),
            "sha256": actual,
            "expected_sha256": expected,
            "identical": expected is None or actual == expected,
        }

    def _build_record(
        self,
        index: int,
        *,
        base_seed: int,
        split: str,
        forced_relation: str | None,
    ) -> SceneRecord:
        sample_seed = _derived_seed(base_seed, index)
        rng = random.Random(sample_seed)
        question_type = QUESTION_TYPES[index % len(QUESTION_TYPES)]
        if question_type == "multi_hop":
            available = [
                depth
                for depth in (2, 3, 4)
                if self.config.min_hops <= depth <= self.config.max_hops
            ]
            if not available:
                question_type = "relation_reasoning"
                hop_depth = 1
            else:
                hop_depth = available[(index // len(QUESTION_TYPES)) % len(available)]
        else:
            hop_depth = 1
        if forced_relation is not None:
            relation = forced_relation
            if question_type == "multi_hop" and relation not in MULTIHOP_RELATIONS:
                question_type = "relation_reasoning"
                hop_depth = 1
        elif question_type == "multi_hop":
            relation = MULTIHOP_RELATIONS[(index // len(QUESTION_TYPES)) % len(MULTIHOP_RELATIONS)]
        else:
            relation = RELATIONS[
                ((index // len(QUESTION_TYPES)) * 3 + index % len(QUESTION_TYPES))
                % len(RELATIONS)
            ]
        objects = self._objects(rng, hop_depth + 1)
        relations = tuple(
            RelationSpec(objects[position].id, relation, objects[position + 1].id).to_dict()
            for position in range(hop_depth)
        )
        object_payloads = tuple(item.to_dict() for item in objects)
        graph = SceneGraph(object_payloads, relations)
        attribute = ("color", "size", "texture")[(index // len(QUESTION_TYPES)) % 3]
        question = generate_question(graph, question_type=question_type, attribute=attribute)
        facts = canonical_facts(object_payloads, relations)
        facts_digest = facts_sha256(facts)
        world_payload = {
            "objects": object_payloads,
            "relations": relations,
            "seed": sample_seed,
        }
        world_digest = hashlib.sha256(
            json.dumps(world_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        scene_id = f"sw_{base_seed}_{index:06d}"
        attributes = tuple(
            {
                "object_id": item.id,
                "category": item.category,
                "shape": item.shape,
                "color": item.color,
                "size": item.size,
                "texture": item.texture,
            }
            for item in objects
        )
        metadata = {
            "generator": self.version,
            "seed": base_seed,
            "sample_seed": sample_seed,
            "split": split,
            "question_type": question.question_type,
            "hop_depth": question.hop_depth,
            "difficulty": question.hop_depth,
            "primary_relation": relation,
            "query": question.query,
            "world_sha256": world_digest,
            "canonical_facts": list(facts),
            "canonical_facts_sha256": facts_digest,
            "information_control": {
                "graph_facts_sha256": facts_digest,
                "caption_facts_sha256": facts_digest,
                "caption_graph_equivalent": True,
                "caption_generator": "deterministic-template-v2",
            },
            "render": {
                "renderer": self.renderer.version,
                "resolution": self.config.image_size,
                "style": self.config.style,
            },
        }
        return SceneRecord(
            scene_id=scene_id,
            image="",
            objects=object_payloads,
            relations=relations,
            question=question.question,
            answer=question.answer,
            choices=question.choices,
            metadata=metadata,
            object_list=object_list_from_world(object_payloads),
            attributes=attributes,
            caption=caption_from_world(object_payloads, relations),
            split=split,
        )

    @staticmethod
    def _objects(rng: random.Random, count: int) -> tuple[ObjectSpec, ...]:
        if count > len(SHAPES):
            raise ValueError("at most five objects are supported by the current ontology")
        # Independent shuffled streams plus uniqueness keep every reference exact.
        shapes = rng.sample(SHAPES, count)
        colors = rng.sample(COLORS, count)
        categories = [rng.choice(CATEGORIES) for _ in range(count)]
        sizes = [rng.choice(SIZES) for _ in range(count)]
        textures = [rng.choice(TEXTURES) for _ in range(count)]
        return tuple(
            ObjectSpec(
                id=f"obj{index + 1}",
                category=categories[index],
                shape=shapes[index],
                color=colors[index],
                size=sizes[index],
                texture=textures[index],
            )
            for index in range(count)
        )

    def _manifest(self, records: list[SceneRecord], dataset_sha: str) -> dict[str, Any]:
        seeds = sorted({int(row.metadata["seed"]) for row in records})
        return {
            "schema_version": 2,
            "generator": self.version,
            "count": len(records),
            "seed": seeds[0] if len(seeds) == 1 else None,
            "seeds": seeds,
            "deterministic_from_seed": True,
            "caption_graph_equivalent": True,
            "semantic_fact_hash": "sha256-canonical-facts-v2",
            "renderer": {
                "version": self.renderer.version,
                "resolution": self.config.image_size,
                "style": self.config.style,
            },
            "dataset_jsonl_sha256": dataset_sha,
            "statistics": {
                "splits": dict(Counter(row.split for row in records)),
                "question_types": dict(
                    Counter(str(row.metadata["question_type"]) for row in records)
                ),
                "hop_depth": {
                    str(key): value
                    for key, value in sorted(
                        Counter(int(row.metadata["hop_depth"]) for row in records).items()
                    )
                },
                "relations": dict(
                    Counter(str(row.metadata["primary_relation"]) for row in records)
                ),
            },
            "reconstruction": {
                "required_inputs": ["objects", "relations", "render.resolution", "render.style"],
                "per_sample_image_checksum": True,
            },
        }


def write_jsonl(path: Path, records: Iterable[SceneRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema_path = Path(__file__).resolve().parents[4] / "schemas" / "synthetic_scene.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.exists() else None
    validator = jsonschema.Draft202012Validator(schema) if schema is not None else None
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            payload = record.to_dict()
            if validator is not None:
                validator.validate(payload)
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _derived_seed(base_seed: int, index: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{index}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _sha256_file(path: Path) -> str:
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


__all__ = ["GeneratorConfig", "SyntheticWorldGenerator", "write_jsonl"]
