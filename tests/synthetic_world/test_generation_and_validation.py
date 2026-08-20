from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import ValidationError

from recoalign.synthetic_world import GeneratorConfig, SyntheticWorldGenerator
from recoalign.synthetic_world.splits import split_leakage_report
from recoalign.synthetic_world.validation import (
    validate_dataset,
    validate_question,
    validate_sample,
)


def test_world_first_sample_schema_and_bundle(tmp_path: Path) -> None:
    generator = SyntheticWorldGenerator(GeneratorConfig(seed=19, image_size=128))
    records = generator.generate(
        8, output_dir=tmp_path / "dataset", split_strategy="composition"
    )
    report = validate_dataset(
        records, split_strategy="composition", check_images=True
    )
    assert report["valid"] is True
    sample = records[0]
    payload = sample.to_dict()
    assert payload["id"] == sample.scene_id
    assert payload["scene_graph"]["relations"] == payload["relations"]
    assert payload["metadata"]["information_control"]["caption_graph_equivalent"] is True
    bundle = tmp_path / "dataset" / "samples" / sample.scene_id
    assert {"image.png", "scene.json", "graph.json", "metadata.json"} <= {
        path.name for path in bundle.iterdir()
    }
    assert json.loads((bundle / "scene.json").read_text(encoding="utf-8"))["caption"]


def test_image_reproducibility_is_byte_identical(tmp_path: Path) -> None:
    generator = SyntheticWorldGenerator(GeneratorConfig(seed=23, image_size=128))
    first = generator.generate(1, output_dir=tmp_path / "first")[0]
    second = generator.generate(1, output_dir=tmp_path / "second")[0]
    assert first.metadata["image_sha256"] == second.metadata["image_sha256"]
    rebuilt = generator.rebuild_image(first, tmp_path / "rebuilt.png")
    assert rebuilt["identical"] is True


@pytest.mark.parametrize("strategy", ["iid", "composition", "relation", "attribute"])
def test_split_policies_have_no_declared_leakage(strategy: str) -> None:
    generator = SyntheticWorldGenerator(GeneratorConfig(seed=29, write_images=False))
    records = generator.generate(240, split_strategy=strategy)
    report = split_leakage_report(records, strategy)
    assert report["valid"] is True
    assert sum(report["counts"].values()) == 240
    replacement = "test" if records[0].split != "test" else "train"
    tampered = [replace(records[0], split=replacement), *records[1:]]
    assert split_leakage_report(tampered, strategy)["valid"] is False


def test_question_answer_is_recomputed_from_graph() -> None:
    generator = SyntheticWorldGenerator(GeneratorConfig(seed=31, write_images=False))
    records = generator.generate(24)
    assert {record.metadata["question_type"] for record in records} == {
        "object_reasoning",
        "relation_reasoning",
        "multi_hop",
        "attribute_reasoning",
    }
    assert {record.metadata["hop_depth"] for record in records} == {1, 2, 3, 4}
    for record in records:
        assert validate_question(record)["derived_answer"] == record.answer
    corrupted = replace(records[0], answer="behind")
    with pytest.raises(ValueError, match="does not match derived answer"):
        validate_question(corrupted)


def test_ood_composition_does_not_confound_question_family() -> None:
    generator = SyntheticWorldGenerator(GeneratorConfig(seed=37, write_images=False))
    train, test = generator.generate_ood_splits(24, 24)
    train_types = {str(record.metadata["question_type"]) for record in train}
    test_types = {str(record.metadata["question_type"]) for record in test}
    assert train_types == test_types == {
        "object_reasoning",
        "relation_reasoning",
        "multi_hop",
        "attribute_reasoning",
    }


def test_schema_rejects_information_control_tampering() -> None:
    record = SyntheticWorldGenerator(GeneratorConfig(write_images=False)).generate(1)[0]
    metadata = dict(record.metadata)
    metadata["information_control"] = {
        **metadata["information_control"],
        "caption_graph_equivalent": False,
    }
    with pytest.raises(ValidationError):
        validate_sample(replace(record, metadata=metadata))
