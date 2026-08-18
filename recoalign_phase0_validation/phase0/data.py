from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from tqdm import tqdm

from .config import ExperimentConfig

OBJECT_PAIRS = (
    ("circle", "square"),
    ("square", "triangle"),
    ("triangle", "circle"),
)
ATTRIBUTES = ("red", "blue", "green")
ATTRIBUTE_PAIRS = tuple(permutations(ATTRIBUTES, 2))
RELATIONS = ("left", "right", "above", "below")
COLORS = {
    "red": (220, 55, 55),
    "blue": (45, 105, 210),
    "green": (45, 165, 85),
}
INVERSE_RELATION = {"left": "right", "right": "left", "above": "below", "below": "above"}


@dataclass(frozen=True)
class Style:
    background: tuple[int, int, int]
    size1: int
    size2: int
    jitter_x: int
    jitter_y: int
    outline_width: int


def _composition_classes() -> list[tuple[str, str, str, str, str]]:
    return [
        (object1, object2, attribute1, attribute2, relation)
        for (object1, object2), (attribute1, attribute2), relation in product(
            OBJECT_PAIRS, ATTRIBUTE_PAIRS, RELATIONS
        )
    ]


def _make_style(rng: random.Random) -> Style:
    shade = rng.randint(244, 255)
    return Style(
        background=(shade, shade, min(255, shade + rng.randint(0, 3))),
        size1=rng.randint(25, 32),
        size2=rng.randint(25, 32),
        jitter_x=rng.randint(-7, 7),
        jitter_y=rng.randint(-7, 7),
        outline_width=rng.randint(2, 4),
    )


def _centers(
    relation: str, image_size: int, style: Style
) -> tuple[tuple[int, int], tuple[int, int]]:
    lo = int(image_size * 0.29)
    hi = int(image_size * 0.71)
    mid = image_size // 2
    jx, jy = style.jitter_x, style.jitter_y
    if relation == "left":
        return (lo + jx, mid + jy), (hi + jx, mid + jy)
    if relation == "right":
        return (hi + jx, mid + jy), (lo + jx, mid + jy)
    if relation == "above":
        return (mid + jx, lo + jy), (mid + jx, hi + jy)
    if relation == "below":
        return (mid + jx, hi + jy), (mid + jx, lo + jy)
    raise ValueError(f"Unknown relation: {relation}")


def _draw_shape(
    draw: ImageDraw.ImageDraw,
    shape: str,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    outline_width: int,
) -> None:
    cx, cy = center
    box = (cx - radius, cy - radius, cx + radius, cy + radius)
    outline = (35, 35, 35)
    if shape == "circle":
        draw.ellipse(box, fill=color, outline=outline, width=outline_width)
    elif shape == "square":
        draw.rounded_rectangle(
            box, radius=max(2, radius // 8), fill=color, outline=outline, width=outline_width
        )
    elif shape == "triangle":
        points = ((cx, cy - radius), (cx - radius, cy + radius), (cx + radius, cy + radius))
        draw.polygon(points, fill=color, outline=outline)
        if outline_width > 1:
            draw.line((*points, points[0]), fill=outline, width=outline_width, joint="curve")
    else:
        raise ValueError(f"Unknown shape: {shape}")


def render_scene(
    path: Path,
    *,
    object1: str,
    object2: str,
    attribute1: str,
    attribute2: str,
    relation: str,
    image_size: int,
    style: Style,
) -> None:
    # Draw at 2x and downsample to avoid jagged edges without introducing a
    # rendering dependency beyond Pillow.
    scale = 2
    canvas = Image.new("RGB", (image_size * scale, image_size * scale), style.background)
    draw = ImageDraw.Draw(canvas)
    center1, center2 = _centers(relation, image_size, style)
    _draw_shape(
        draw,
        object1,
        (center1[0] * scale, center1[1] * scale),
        style.size1 * scale,
        COLORS[attribute1],
        style.outline_width * scale,
    )
    _draw_shape(
        draw,
        object2,
        (center2[0] * scale, center2[1] * scale),
        style.size2 * scale,
        COLORS[attribute2],
        style.outline_width * scale,
    )
    canvas.resize((image_size, image_size), Image.Resampling.LANCZOS).save(path, format="PNG")


def _metadata(
    *,
    image_id: str,
    relative_path: str,
    split: str,
    object1: str,
    object2: str,
    attribute1: str,
    attribute2: str,
    relation: str,
    pair_id: str | None = None,
) -> dict[str, Any]:
    composition = f"a {attribute1} {object1} is {relation} of a {attribute2} {object2}"
    record: dict[str, Any] = {
        "image_id": image_id,
        "image": relative_path.replace("\\", "/"),
        "split": split,
        "object1": object1,
        "object2": object2,
        "attribute1": attribute1,
        "attribute2": attribute2,
        "relation": relation,
        "composition": composition,
        # Probe labels are explicit to prevent silent changes in factor semantics.
        "object_label": f"{object1}|{object2}",
        "attribute_label": f"{attribute1}|{attribute2}",
        "semantic_label": {
            "object": f"{object1}|{object2}",
            "attribute": f"{attribute1}|{attribute2}",
            "relation": relation,
            "composition": composition,
        },
    }
    if pair_id is not None:
        record["pair_id"] = pair_id
    return record


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _balanced_assignments(n: int, seed: int) -> list[tuple[str, str, str, str, str]]:
    classes = _composition_classes()
    assignments = [classes[index % len(classes)] for index in range(n)]
    random.Random(seed).shuffle(assignments)
    return assignments


def _manifest_matches(config: ExperimentConfig) -> bool:
    path = config.dataset_dir / "manifest.json"
    if not path.exists():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "seed": config.seed,
        "n_train": config.n_train,
        "n_test": config.n_test,
        "n_control_pairs": config.n_control_pairs,
        "image_size": config.image_size,
        "generator_version": 1,
    }
    return all(manifest.get(key) == value for key, value in expected.items())


def generate_dataset(config: ExperimentConfig) -> Path:
    config.validate()
    metadata_path = config.dataset_dir / "metadata.jsonl"
    control_path = config.dataset_dir / "control_metadata.jsonl"
    if (
        not config.force_data
        and _manifest_matches(config)
        and metadata_path.exists()
        and control_path.exists()
    ):
        return metadata_path

    image_root = config.dataset_dir / "images"
    control_root = config.dataset_dir / "control_images"
    image_root.mkdir(parents=True, exist_ok=True)
    control_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    split_specs = (
        ("train", config.n_train, config.seed + 11),
        ("test", config.n_test, config.seed + 29),
    )
    for split, count, seed in split_specs:
        assignments = _balanced_assignments(count, seed)
        split_dir = image_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for index, values in enumerate(tqdm(assignments, desc=f"Generate {split}")):
            object1, object2, attribute1, attribute2, relation = values
            image_id = f"{split}_{index:06d}"
            path = split_dir / f"{image_id}.png"
            style = _make_style(random.Random(seed * 1_000_003 + index))
            render_scene(
                path,
                object1=object1,
                object2=object2,
                attribute1=attribute1,
                attribute2=attribute2,
                relation=relation,
                image_size=config.image_size,
                style=style,
            )
            records.append(
                _metadata(
                    image_id=image_id,
                    relative_path=str(path.relative_to(config.dataset_dir)),
                    split=split,
                    object1=object1,
                    object2=object2,
                    attribute1=attribute1,
                    attribute2=attribute2,
                    relation=relation,
                )
            )
    _write_jsonl(metadata_path, records)

    control_records: list[dict[str, Any]] = []
    classes = _composition_classes()
    for pair_index in tqdm(range(config.n_control_pairs), desc="Generate isolation control"):
        object1, object2, attribute1, attribute2, _ = classes[pair_index % len(classes)]
        first = "left" if pair_index % 2 == 0 else "above"
        second = INVERSE_RELATION[first]
        pair_id = f"pair_{pair_index:05d}"
        style = _make_style(random.Random(config.seed * 2_000_003 + pair_index))
        for member, relation in (("A", first), ("B", second)):
            image_id = f"{pair_id}_{member}"
            path = control_root / f"{image_id}.png"
            render_scene(
                path,
                object1=object1,
                object2=object2,
                attribute1=attribute1,
                attribute2=attribute2,
                relation=relation,
                image_size=config.image_size,
                style=style,
            )
            control_records.append(
                _metadata(
                    image_id=image_id,
                    relative_path=str(path.relative_to(config.dataset_dir)),
                    split="control",
                    object1=object1,
                    object2=object2,
                    attribute1=attribute1,
                    attribute2=attribute2,
                    relation=relation,
                    pair_id=pair_id,
                )
            )
    _write_jsonl(control_path, control_records)

    metadata_sha256 = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    control_sha256 = hashlib.sha256(control_path.read_bytes()).hexdigest()
    manifest = {
        "generator_version": 1,
        "seed": config.seed,
        "n_train": config.n_train,
        "n_test": config.n_test,
        "n_control_pairs": config.n_control_pairs,
        "n_control_images": 2 * config.n_control_pairs,
        "image_size": config.image_size,
        "composition_classes": len(_composition_classes()),
        "metadata_sha256": metadata_sha256,
        "control_metadata_sha256": control_sha256,
    }
    (config.dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
