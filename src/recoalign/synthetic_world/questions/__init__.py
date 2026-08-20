"""Controlled compositional question templates."""

from .conditions import CONDITIONS, Condition, condition_image, condition_text, graph_variant
from .generation import (
    QUESTION_TYPES,
    QuestionSpec,
    canonical_facts,
    caption_from_world,
    facts_sha256,
    generate_question,
    object_description,
    object_list_from_world,
)

__all__ = [
    "QUESTION_TYPES",
    "CONDITIONS",
    "Condition",
    "QuestionSpec",
    "canonical_facts",
    "caption_from_world",
    "condition_image",
    "condition_text",
    "facts_sha256",
    "generate_question",
    "graph_variant",
    "object_description",
    "object_list_from_world",
]
