"""Content-addressed inference cache for deterministic VLM generations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class InferenceCache:
    def __init__(self, root: str | Path | None, *, enabled: bool = True) -> None:
        self.root = Path(root) if root else None
        self.enabled = enabled and self.root is not None

    @staticmethod
    def key(payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        path = self._path(key)
        if path is None or not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return str(payload["prediction"])

    def put(self, key: str, prediction: str, metadata: dict[str, Any]) -> None:
        path = self._path(key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {"prediction": prediction, "metadata": metadata},
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)

    def _path(self, key: str) -> Path | None:
        if not self.enabled or self.root is None:
            return None
        return self.root / key[:2] / f"{key}.json"


__all__ = ["InferenceCache"]
