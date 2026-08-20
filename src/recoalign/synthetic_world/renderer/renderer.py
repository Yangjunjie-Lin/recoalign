"""Deterministic rasterization of a declared synthetic world state."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from recoalign.synthetic_world.ontology import normalize_relation

RGB = tuple[int, int, int]

BASE_COLORS: dict[str, RGB] = {
    "red": (211, 63, 73),
    "blue": (63, 103, 210),
    "green": (50, 158, 91),
    "yellow": (226, 177, 48),
    "purple": (139, 78, 181),
}
SIZE_SCALE = {"small": 0.72, "medium": 0.9, "large": 1.08}
STYLES = ("flat", "outline", "pastel")


@dataclass(frozen=True)
class RendererConfig:
    resolution: int = 192
    style: str = "flat"
    background: RGB = (247, 248, 251)

    def __post_init__(self) -> None:
        if self.resolution < 96:
            raise ValueError("renderer resolution must be at least 96")
        if self.style not in STYLES:
            raise ValueError(f"renderer style must be one of {STYLES}")


class DeterministicRenderer:
    """Render only from explicit objects, relations, resolution, and style."""

    version = "recoalign.synthetic_renderer.v2"

    def __init__(self, config: RendererConfig | None = None) -> None:
        self.config = config or RendererConfig()

    def render(
        self,
        path: str | Path,
        objects: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        relations: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        size = self.config.resolution
        background = self._style_color(self.config.background)
        image = Image.new("RGB", (size, size), background)
        draw = ImageDraw.Draw(image)
        positions, depth, containment = self._layout(objects, relations)
        primary = normalize_relation(dict(relations[0]))["relation"] if relations else "left"
        ordered = sorted(objects, key=lambda node: depth.get(str(node["id"]), 0))
        if primary in {"touching", "holding"} and len(objects) >= 2:
            first = positions[str(objects[0]["id"])]
            second = positions[str(objects[1]["id"])]
            draw.line((*first, *second), fill=(77, 82, 94), width=max(2, size // 80))
        for node in ordered:
            identifier = str(node["id"])
            radius = self._radius(node, depth.get(identifier, 0), containment.get(identifier))
            self._draw_object(image, node, positions[identifier], radius)
        if primary == "holding" and len(objects) >= 2:
            first = positions[str(objects[0]["id"])]
            second = positions[str(objects[1]["id"])]
            handle = max(3, size // 48)
            draw.arc(
                (second[0] - handle, second[1] - handle, second[0] + handle, second[1] + handle),
                30,
                330,
                fill=(30, 35, 45),
                width=2,
            )
        image.save(destination, format="PNG", optimize=False, compress_level=9)
        return destination

    def image_sha256(self, path: str | Path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def _layout(
        self,
        objects: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        relations: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> tuple[dict[str, tuple[int, int]], dict[str, int], dict[str, str]]:
        size = self.config.resolution
        margin = int(size * 0.16)
        center = size // 2
        count = max(1, len(objects))
        step = (size - 2 * margin) / max(1, count - 1)
        relation = normalize_relation(dict(relations[0]))["relation"] if relations else "left"
        positions: dict[str, tuple[int, int]] = {}
        depth: dict[str, int] = {}
        containment: dict[str, str] = {}
        for index, node in enumerate(objects):
            identifier = str(node["id"])
            x = int(round(margin + step * index))
            y = center + (index % 2) * max(2, size // 28) - max(1, size // 56)
            if relation == "right":
                x = size - x
            elif relation == "above":
                x, y = center + (index % 2) * size // 28, int(round(margin + step * index))
            elif relation == "below":
                x, y = center + (index % 2) * size // 28, size - int(round(margin + step * index))
            elif relation in {"front", "behind"}:
                x = int(round(margin + step * index))
                y = int(round(margin + step * index))
                depth[identifier] = (count - index) if relation == "front" else index + 1
            elif relation == "near":
                x = center + (index * 2 - 1) * size // 10
                y = center
            elif relation == "far":
                x = margin if index == 0 else size - margin
                y = center
            elif relation in {"inside", "contains"}:
                x, y = center, center
                containment[identifier] = (
                    "inner"
                    if (relation == "inside" and index == 0)
                    or (relation == "contains" and index > 0)
                    else "outer"
                )
                depth[identifier] = 2 if containment[identifier] == "inner" else 0
            elif relation == "touching":
                x = center + (-1 if index == 0 else 1) * size // 9
                y = center
            elif relation == "holding":
                x = center + (-1 if index == 0 else 1) * size // 8
                y = center + (0 if index == 0 else size // 12)
            positions[identifier] = (x, y)
            depth.setdefault(identifier, index)
        return positions, depth, containment

    def _radius(self, node: dict[str, Any], depth: int, containment: str | None) -> int:
        base = self.config.resolution * 0.078 * SIZE_SCALE[str(node["size"])]
        if containment == "outer":
            base *= 2.1
        elif containment == "inner":
            base *= 0.62
        if depth and containment is None:
            base *= 0.88 + min(depth, 5) * 0.035
        return max(7, int(round(base)))

    def _draw_object(
        self,
        image: Image.Image,
        node: dict[str, Any],
        center: tuple[int, int],
        radius: int,
    ) -> None:
        mask = Image.new("L", image.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        cx, cy = center
        box = (cx - radius, cy - radius, cx + radius, cy + radius)
        shape = str(node["shape"])
        if shape in {"circle", "sphere"}:
            mask_draw.ellipse(box, fill=255)
        elif shape == "triangle":
            mask_draw.polygon(
                [(cx, cy - radius), (cx - radius, cy + radius), (cx + radius, cy + radius)],
                fill=255,
            )
        elif shape == "cube":
            offset = max(3, radius // 3)
            mask_draw.polygon(
                [
                    (cx - radius, cy - radius + offset),
                    (cx, cy - radius),
                    (cx + radius, cy - radius + offset),
                    (cx + radius, cy + radius - offset),
                    (cx, cy + radius),
                    (cx - radius, cy + radius - offset),
                ],
                fill=255,
            )
        else:
            mask_draw.rectangle(box, fill=255)
        fill = self._style_color(BASE_COLORS[str(node["color"])])
        layer = Image.new("RGB", image.size, fill)
        texture_draw = ImageDraw.Draw(layer)
        texture = str(node["texture"])
        accent = tuple(max(0, value - 48) for value in fill)
        if texture == "striped":
            spacing = max(4, radius // 3)
            for offset in range(-2 * radius, 2 * radius + 1, spacing):
                texture_draw.line(
                    (cx - radius, cy + offset, cx + radius, cy + offset - radius),
                    fill=accent,
                    width=max(1, radius // 9),
                )
        elif texture == "dotted":
            dot = max(1, radius // 8)
            spacing = max(5, radius // 2)
            for x in range(cx - radius, cx + radius + 1, spacing):
                for y in range(cy - radius, cy + radius + 1, spacing):
                    texture_draw.ellipse((x - dot, y - dot, x + dot, y + dot), fill=accent)
        image.paste(layer, mask=mask)
        draw = ImageDraw.Draw(image)
        outline = (31, 36, 47)
        width = 3 if self.config.style == "outline" else 2
        if shape in {"circle", "sphere"}:
            draw.ellipse(box, outline=outline, width=width)
            if shape == "sphere":
                highlight = max(2, radius // 4)
                draw.ellipse(
                    (
                        cx - radius // 2,
                        cy - radius // 2,
                        cx - radius // 2 + highlight,
                        cy - radius // 2 + highlight,
                    ),
                    fill=(245, 245, 245),
                )
        elif shape == "triangle":
            draw.line(
                [
                    (cx, cy - radius),
                    (cx - radius, cy + radius),
                    (cx + radius, cy + radius),
                    (cx, cy - radius),
                ],
                fill=outline,
                width=width,
            )
        elif shape == "cube":
            offset = max(3, radius // 3)
            polygon = [
                (cx - radius, cy - radius + offset),
                (cx, cy - radius),
                (cx + radius, cy - radius + offset),
                (cx + radius, cy + radius - offset),
                (cx, cy + radius),
                (cx - radius, cy + radius - offset),
                (cx - radius, cy - radius + offset),
            ]
            draw.line(polygon, fill=outline, width=width)
            draw.line((cx, cy - radius, cx, cy + radius), fill=outline, width=1)
        else:
            draw.rectangle(box, outline=outline, width=width)
        self._draw_category_marker(draw, node, center, radius)

    @staticmethod
    def _draw_category_marker(
        draw: ImageDraw.ImageDraw,
        node: dict[str, Any],
        center: tuple[int, int],
        radius: int,
    ) -> None:
        cx, cy = center
        category = str(node["category"])
        marker = (25, 29, 38)
        if category == "animal":
            ear = max(2, radius // 4)
            draw.polygon(
                [(cx - ear, cy - radius), (cx, cy - radius - ear), (cx, cy - radius)],
                fill=marker,
            )
            draw.polygon(
                [(cx, cy - radius), (cx + ear, cy - radius - ear), (cx + ear, cy - radius)],
                fill=marker,
            )
        elif category == "vehicle":
            wheel = max(2, radius // 6)
            draw.ellipse(
                (
                    cx - radius // 2 - wheel,
                    cy + radius - wheel,
                    cx - radius // 2 + wheel,
                    cy + radius + wheel,
                ),
                fill=marker,
            )
            draw.ellipse(
                (
                    cx + radius // 2 - wheel,
                    cy + radius - wheel,
                    cx + radius // 2 + wheel,
                    cy + radius + wheel,
                ),
                fill=marker,
            )

    def _style_color(self, color: RGB) -> RGB:
        if self.config.style == "pastel":
            return tuple(int(round(value * 0.62 + 255 * 0.38)) for value in color)
        if self.config.style == "outline":
            return tuple(int(round(value * 0.42 + 255 * 0.58)) for value in color)
        return color


__all__ = ["DeterministicRenderer", "RendererConfig", "STYLES"]
