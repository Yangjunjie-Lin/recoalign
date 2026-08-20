"""Versioned prompting protocol shared by every VLM backbone."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import yaml


@dataclass(frozen=True)
class PromptProtocol:
    """A model-independent scientific prompt, excluding adapter chat wrappers."""

    protocol_id: str = "reasoning-default-v1"
    instruction: str = ""
    question_template: str = "Question: {question}"
    answer_template: str = "Answer with exactly one option: {choices}."
    separator: str = "\n"
    temperature: float = 0.0
    do_sample: bool = False
    max_new_tokens: int = 32
    source: str | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.protocol_id:
            raise ValueError("prompt protocol_id must be non-empty")
        if "{question}" not in self.question_template:
            raise ValueError("question_template must contain {question}")
        if "{choices}" not in self.answer_template:
            raise ValueError("answer_template must contain {choices}")
        if self.temperature < 0.0:
            raise ValueError("prompt temperature must be non-negative")
        if self.max_new_tokens <= 0:
            raise ValueError("prompt max_new_tokens must be positive")
        if not self.do_sample and self.temperature != 0.0:
            raise ValueError("deterministic prompting requires temperature=0 when do_sample=false")

    def render_question(self, question: str, choices: tuple[str, ...]) -> str:
        rendered = self.question_template.format(question=question)
        answer = self.answer_template.format(choices=", ".join(choices))
        return self.separator.join((rendered, answer))

    def render(self, evidence: str, question: str, choices: tuple[str, ...]) -> str:
        parts = [value for value in (self.instruction, evidence) if value]
        parts.append(self.render_question(question, choices))
        return self.separator.join(parts)

    def generation_config(self) -> dict[str, Any]:
        return {
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "instruction": self.instruction,
            "question_template": self.question_template,
            "answer_template": self.answer_template,
            "separator": self.separator,
            "generation": self.generation_config(),
            "source": self.source,
            "sha256": self.sha256,
        }


def load_prompt_protocol(path: str | Path | None) -> PromptProtocol:
    """Load and hash a protocol, or return the exact Phase-1 prompt as the default."""

    if path is None:
        payload = _default_payload()
        digest = _payload_digest(payload)
        return _from_payload(payload, source=None, digest=digest)
    source = Path(path)
    if not source.is_absolute():
        source = _repository_root() / source
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"prompt protocol must be a mapping: {source}")
    schema = json.loads(
        (_repository_root() / "schemas" / "prompt_protocol.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(payload, schema)
    if payload.get("inherits"):
        parent = source.parent / str(payload["inherits"])
        parent_payload = yaml.safe_load(parent.read_text(encoding="utf-8"))
        if not isinstance(parent_payload, dict):
            raise ValueError(f"parent prompt protocol must be a mapping: {parent}")
        payload = _merge_prompt(parent_payload, payload)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return _from_payload(payload, source=source.as_posix(), digest=digest)


def _from_payload(
    payload: dict[str, Any], *, source: str | None, digest: str
) -> PromptProtocol:
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("prompt protocol schema_version must be 1")
    generation = payload.get("generation", {})
    if not isinstance(generation, dict):
        raise ValueError("prompt generation configuration must be a mapping")
    return PromptProtocol(
        protocol_id=str(payload["protocol_id"]),
        instruction=str(payload.get("instruction", "")),
        question_template=str(payload["question_template"]),
        answer_template=str(payload["answer_template"]),
        separator=str(payload.get("separator", "\n")),
        temperature=float(generation.get("temperature", 0.0)),
        do_sample=bool(generation.get("do_sample", False)),
        max_new_tokens=int(generation.get("max_new_tokens", 32)),
        source=source,
        sha256=digest,
    )


def _merge_prompt(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    merged = dict(parent)
    merged.update({key: value for key, value in child.items() if key != "inherits"})
    generation = dict(parent.get("generation", {}))
    generation.update(dict(child.get("generation", {})))
    merged["generation"] = generation
    return merged


def _default_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": "reasoning-default-v1",
        "instruction": "",
        "question_template": "Question: {question}",
        "answer_template": "Answer with exactly one option: {choices}.",
        "separator": "\n",
        "generation": {"temperature": 0.0, "do_sample": False, "max_new_tokens": 32},
    }


def _payload_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


__all__ = ["PromptProtocol", "load_prompt_protocol"]
