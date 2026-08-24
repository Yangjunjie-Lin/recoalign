"""Deterministic M2 visual-legend and context composition."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from datasets.records import SceneRecord
from recoalign.synthetic_world.renderer import DeterministicRenderer, RendererConfig

from .scaffold_comprehension import ConstructScaffold

NEUTRAL_RGB = (229, 232, 238)
TILE_BACKGROUND = (245, 245, 245)
CONTEXT_SIZE = 192
TILE_SIZE = 96
LEGEND_CANVAS_SIZE = (576, 384)


def create_neutral_image(path: str | Path) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (CONTEXT_SIZE, CONTEXT_SIZE), NEUTRAL_RGB)
    image.save(destination, format="PNG", optimize=False, compress_level=9)
    return image_metadata(destination)


def render_visual_legend(
    record: SceneRecord,
    scaffold: ConstructScaffold,
    *,
    output_path: str | Path,
    tile_dir: str | Path,
) -> dict[str, Any]:
    if scaffold.manipulation != "M2":
        raise ValueError("visual legends are registered only for M2")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tile_root = Path(tile_dir)
    tile_root.mkdir(parents=True, exist_ok=True)
    objects = {str(node["id"]): dict(node) for node in record.objects}
    renderer = DeterministicRenderer(
        RendererConfig(resolution=TILE_SIZE, style="flat", background=TILE_BACKGROUND)
    )
    by_entity = scaffold.by_entity()
    canvas = Image.new("RGB", LEGEND_CANVAS_SIZE, (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row_index, entity_id in enumerate(scaffold.row_order):
        binding = by_entity[entity_id]
        tile_path = tile_root / f"{record.scene_id}_{binding.source_object_id}.png"
        if not tile_path.is_file():
            renderer.render(tile_path, [objects[binding.source_object_id]], [])
        y = row_index * TILE_SIZE
        with Image.open(tile_path) as source:
            canvas.paste(source.convert("RGB"), (CONTEXT_SIZE, y))
        draw.rectangle(
            (CONTEXT_SIZE, y, LEGEND_CANVAS_SIZE[0] - 1, y + TILE_SIZE - 1),
            outline=(80, 84, 92),
            width=1,
        )
        draw.text(
            (CONTEXT_SIZE + TILE_SIZE + 12, y + 34),
            f"{entity_id} | shape={binding.shape} | color={binding.color}",
            fill=(24, 27, 34),
            font=font,
        )
    canvas.save(destination, format="PNG", optimize=False, compress_level=9)
    return {
        **image_metadata(destination),
        "scene_id": record.scene_id,
        "evidence_truth": scaffold.evidence_truth,
        "row_order": list(scaffold.row_order),
        "tile_policy": "deterministic_isolated_rerender",
        "tile_size": [TILE_SIZE, TILE_SIZE],
        "absolute_coordinates_preserved": False,
    }


def compose_context_with_legend(
    context_path: str | Path,
    legend_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(legend_path) as legend_source:
        canvas = legend_source.convert("RGB")
    with Image.open(context_path) as context_source:
        context = context_source.convert("RGB").resize((CONTEXT_SIZE, CONTEXT_SIZE))
    y = (LEGEND_CANVAS_SIZE[1] - CONTEXT_SIZE) // 2
    canvas.paste(context, (0, y))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, y, CONTEXT_SIZE - 1, y + CONTEXT_SIZE - 1), outline=(80, 84, 92), width=1)
    canvas.save(destination, format="PNG", optimize=False, compress_level=9)
    return image_metadata(destination)


def build_m2_scene_assets(
    record: SceneRecord,
    scaffolds: tuple[ConstructScaffold, ConstructScaffold],
    *,
    neutral_image: str | Path,
    output_dir: str | Path,
) -> tuple[dict[tuple[str, str, str], str], dict[str, Any]]:
    root = Path(output_dir)
    tile_dir = root / "tiles"
    mapping: dict[tuple[str, str, str], str] = {}
    report: dict[str, Any] = {"scene_id": record.scene_id, "conditions": {}}
    for scaffold in scaffolds:
        condition_root = root / scaffold.evidence_truth
        legend_path = condition_root / "legend.png"
        legend = render_visual_legend(
            record,
            scaffold,
            output_path=legend_path,
            tile_dir=tile_dir,
        )
        context_reports = {}
        for image_context, context_path in (
            ("neutral_image", neutral_image),
            ("original_scene_image", record.image),
        ):
            composite = condition_root / f"{image_context}.png"
            metadata = compose_context_with_legend(context_path, legend_path, composite)
            mapping[
                (record.scene_id, scaffold.evidence_truth, image_context)
            ] = composite.as_posix()
            context_reports[image_context] = metadata
        report["conditions"][scaffold.evidence_truth] = {
            "legend": legend,
            "contexts": context_reports,
        }
    report["tiles"] = [image_metadata(path) for path in sorted(tile_dir.glob("*.png"))]
    oracle = report["conditions"]["oracle"]
    corrupted = report["conditions"]["corrupted"]
    report["assertions"] = {
        "legend_dimensions_equal": (
            oracle["legend"]["dimensions"] == corrupted["legend"]["dimensions"]
        ),
        "neutral_composite_dimensions_equal": (
            oracle["contexts"]["neutral_image"]["dimensions"]
            == corrupted["contexts"]["neutral_image"]["dimensions"]
        ),
        "original_composite_dimensions_equal": (
            oracle["contexts"]["original_scene_image"]["dimensions"]
            == corrupted["contexts"]["original_scene_image"]["dimensions"]
        ),
        "row_order_equal": (
            oracle["legend"]["row_order"] == corrupted["legend"]["row_order"]
        ),
        "absolute_coordinates_preserved": False,
    }
    report["passed"] = all(
        bool(value)
        for key, value in report["assertions"].items()
        if key != "absolute_coordinates_preserved"
    ) and report["assertions"]["absolute_coordinates_preserved"] is False
    report["passed"] = report["passed"] and len(report["tiles"]) == 4
    return mapping, report


def image_metadata(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with Image.open(source) as image:
        dimensions = [int(image.width), int(image.height)]
        mode = image.mode
    return {
        "path": source.as_posix(),
        "bytes": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "dimensions": dimensions,
        "mode": mode,
    }


__all__ = [
    "CONTEXT_SIZE",
    "LEGEND_CANVAS_SIZE",
    "NEUTRAL_RGB",
    "TILE_BACKGROUND",
    "TILE_SIZE",
    "compose_context_with_legend",
    "build_m2_scene_assets",
    "create_neutral_image",
    "image_metadata",
    "render_visual_legend",
]
