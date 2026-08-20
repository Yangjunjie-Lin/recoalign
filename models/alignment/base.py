"""No-op alignment boundary; no novel loss is introduced in Phase 1."""

from __future__ import annotations

from typing import Any, Protocol


class AlignmentModule(Protocol):
    def align(self, visual: Any, structured: Any) -> Any:
        ...


class IdentityAlignment:
    """Reference implementation that leaves representations unchanged."""

    def align(self, visual: Any, structured: Any) -> tuple[Any, Any]:
        return visual, structured
