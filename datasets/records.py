"""Serializable records for synthetic compositional scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SceneRecord:
    """One scene, its graph, and a compositional question/answer pair."""

    scene_id: str
    image: str
    objects: tuple[dict[str, Any], ...]
    relations: tuple[dict[str, Any], ...]
    question: str
    answer: str
    choices: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    object_list: tuple[str, ...] = ()
    attributes: tuple[dict[str, Any], ...] = ()
    caption: str = ""
    split: str = "unspecified"

    def to_dict(self) -> dict[str, Any]:
        object_list = self.object_list or tuple(
            " ".join(
                str(item)
                for item in (
                    node.get("size"),
                    node.get("texture"),
                    node.get("color"),
                    node.get("shape", node.get("category", "object")),
                )
                if item
            )
            for node in self.objects
        )
        attributes = self.attributes or tuple(
            {
                "object_id": str(node["id"]),
                **{
                    key: node[key]
                    for key in ("color", "size", "texture", "category", "shape")
                    if key in node
                },
            }
            for node in self.objects
        )
        canonical_relations = [
            {
                "subject": str(item.get("subject", item.get("source", ""))),
                "relation": str(item["relation"]),
                "object": str(item.get("object", item.get("target", ""))),
            }
            for item in self.relations
        ]
        return {
            "id": self.scene_id,
            "scene_id": self.scene_id,
            "image": self.image,
            "objects": [dict(item) for item in self.objects],
            "object_list": list(object_list),
            "attributes": [dict(item) for item in attributes],
            "relations": canonical_relations,
            "scene_graph": {
                "objects": [dict(item) for item in self.objects],
                "relations": canonical_relations,
            },
            "graph": {
                "nodes": [dict(item) for item in self.objects],
                "edges": canonical_relations,
            },
            "caption": self.caption,
            "question": self.question,
            "answer": self.answer,
            "choices": list(self.choices),
            "split": self.split,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SceneRecord:
        graph = payload.get("scene_graph") or payload.get("graph") or {}
        objects = payload.get("objects", graph.get("objects", graph.get("nodes", [])))
        relations = payload.get("relations", graph.get("relations", graph.get("edges", [])))
        choices = payload.get("choices", [])
        if not isinstance(objects, list) or not isinstance(relations, list):
            raise ValueError("scene objects and relations must be lists")
        if not isinstance(choices, list) or not choices:
            raise ValueError("scene choices must be a non-empty list")
        return cls(
            scene_id=str(payload.get("id", payload.get("scene_id", ""))),
            image=str(payload.get("image", "")),
            objects=tuple(dict(item) for item in objects),
            relations=tuple(dict(item) for item in relations),
            question=str(payload["question"]),
            answer=str(payload["answer"]),
            choices=tuple(str(item) for item in choices),
            metadata=dict(payload.get("metadata", {})),
            object_list=tuple(str(item) for item in payload.get("object_list", [])),
            attributes=tuple(dict(item) for item in payload.get("attributes", [])),
            caption=str(payload.get("caption", "")),
            split=str(
                payload.get(
                    "split", payload.get("metadata", {}).get("split", "unspecified")
                )
            ),
        )

    @property
    def image_path(self) -> Path | None:
        return Path(self.image) if self.image else None
