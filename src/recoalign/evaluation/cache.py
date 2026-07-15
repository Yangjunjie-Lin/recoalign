"""Safe, deterministic NumPy embedding caches for repeated baseline evaluation."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CacheResult:
    embeddings: np.ndarray
    hit: bool
    path: Path


class EmbeddingCache:
    """Store embeddings without pickle and reject stale identifier orderings."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def get_or_compute(
        self,
        *,
        namespace: str,
        metadata: dict[str, Any],
        identifiers: list[str],
        compute: Any,
        enabled: bool = True,
    ) -> CacheResult:
        key = cache_digest({"namespace": namespace, **metadata})
        path = self.root / f"{namespace}-{key[:16]}.npz"
        if enabled and path.is_file():
            loaded = self._load(path, identifiers)
            if loaded is not None:
                return CacheResult(loaded, True, path)

        embeddings = np.asarray(compute(), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(identifiers):
            raise ValueError(
                f"encoder returned shape {embeddings.shape}, expected [{len(identifiers)}, D]"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("encoder returned non-finite embeddings")
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix=f".{path.stem}-",
                suffix=".npz",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            try:
                np.savez(
                    temporary,
                    embeddings=embeddings,
                    identifiers=np.asarray(identifiers, dtype=np.str_),
                    metadata=np.asarray(
                        json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                        dtype=np.str_,
                    ),
                )
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return CacheResult(embeddings, False, path)

    @staticmethod
    def _load(path: Path, identifiers: list[str]) -> np.ndarray | None:
        try:
            with np.load(path, allow_pickle=False) as payload:
                cached_ids = payload["identifiers"].astype(str).tolist()
                embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        except (OSError, KeyError, ValueError):
            return None
        if cached_ids != identifiers:
            return None
        if embeddings.ndim != 2 or embeddings.shape[0] != len(identifiers):
            return None
        if not np.isfinite(embeddings).all():
            return None
        return embeddings


def cache_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
