"""Visual → structure → reasoning contracts without implementing a new model yet."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class VisualRepresentation:
    values: Any
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StructuredRepresentation:
    nodes: tuple[dict[str, Any], ...]
    edges: tuple[dict[str, Any], ...]
    source: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReasoningRequest:
    question: str
    visual: VisualRepresentation | None = None
    structure: StructuredRepresentation | None = None
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReasoningResult:
    answer: str
    confidence: float | None = None
    rationale: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class StructureEncoder(Protocol):
    def encode(self, visual: VisualRepresentation) -> StructuredRepresentation:
        ...


class LLMReasoning(Protocol):
    def reason(self, request: ReasoningRequest) -> ReasoningResult:
        ...
