"""Compute mechanism-level contrasts without claiming a unique internal bottleneck."""

from __future__ import annotations

from typing import Any

from evaluation.metrics import paired_difference


def summarize_interface_gap(
    rows: list[dict[str, Any]], *, bootstrap_samples: int = 1000, seed: int = 7
) -> dict[str, Any]:
    def compare(first: str, second: str) -> dict[str, Any]:
        return paired_difference(
            rows,
            first,
            second,
            bootstrap_samples=bootstrap_samples,
            seed=seed + len(first) + len(second),
        )

    contrasts = {
        "structured_reasoning_gain": compare("scene_graph", "image_only"),
        "caption_gain": compare("caption", "image_only"),
        "graph_over_text": compare("scene_graph", "caption"),
        "graph_over_objects": compare("scene_graph", "object_list"),
        "full_over_partial": compare("scene_graph", "partial_graph"),
        "full_over_random": compare("scene_graph", "random_graph"),
        "partial_graph_effect": compare("partial_graph", "scene_graph"),
        "corruption_effect": compare("corrupted_graph", "scene_graph"),
    }
    return {
        "hypothesis": "Structured Reasoning Interface Gap",
        "contrasts": contrasts,
        "interpretation": (
            "Explicit structure is a sufficient intervention in the tested interface; this result "
            "does not establish that structure construction is the only internal bottleneck."
        ),
    }
