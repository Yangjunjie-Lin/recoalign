"""Future trainer protocol; no trainable ReCoAlign method is claimed yet."""

from typing import Any, Protocol


class Trainer(Protocol):
    def fit(self, dataset: Any, *, seed: int) -> dict[str, Any]:
        ...
