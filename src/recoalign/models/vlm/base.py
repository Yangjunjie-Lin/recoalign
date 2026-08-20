"""Model-neutral lifecycle, prompting, generation, and evaluation contracts."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from datasets.records import SceneRecord
from recoalign.models.vlm.cache import InferenceCache
from recoalign.models.vlm.evaluation import AnswerEvaluator, EvaluationResult
from recoalign.models.vlm.prompting import PromptProtocol, load_prompt_protocol
from synthetic_world.compositional_tasks.tasks import (
    Condition,
    InputSetting,
    condition_image,
    controlled_evidence_view,
    lexical_token_count,
)


@dataclass(frozen=True)
class PreparedInput:
    prompt: str
    image: str | Path | None
    condition: str
    setting: str
    input_tokens: int
    evidence_tokens: int
    semantic_units: int
    relation_count: int
    semantic_facts_sha256: str | None
    serialization: str
    padding_units: int
    token_match_delta: int
    prompt_protocol_id: str = "reasoning-default-v1"
    prompt_protocol_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "setting": self.setting,
            "input_tokens": self.input_tokens,
            "evidence_tokens": self.evidence_tokens,
            "semantic_units": self.semantic_units,
            "relation_count": self.relation_count,
            "semantic_facts_sha256": self.semantic_facts_sha256,
            "serialization": self.serialization,
            "padding_units": self.padding_units,
            "token_match_delta": self.token_match_delta,
            "prompt_protocol_id": self.prompt_protocol_id,
            "prompt_protocol_sha256": self.prompt_protocol_sha256,
        }


class BaseVLM(ABC):
    """Shared API for Reference, LLaVA, Qwen-VL, and InternVL backbones."""

    model_id = "base-vlm"

    def __init__(
        self,
        *,
        prompt_protocol: PromptProtocol | None = None,
        evaluator: AnswerEvaluator | None = None,
        cache: InferenceCache | None = None,
        batch_size: int = 1,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("VLM batch_size must be positive")
        self.prompt_protocol = prompt_protocol or load_prompt_protocol(None)
        self.answer_evaluator = evaluator or AnswerEvaluator()
        self.inference_cache = cache or InferenceCache(None, enabled=False)
        self.batch_size = batch_size
        self._loaded = False

    @abstractmethod
    def load(self) -> BaseVLM:
        """Load model resources once and return this instance."""

    @abstractmethod
    def generate(self, prompt: str, *, image: str | Path | None = None, **kwargs: Any) -> str:
        """Generate one answer from the shared scientific prompt."""

    def generate_with_interface(
        self,
        prompt: str,
        *,
        image: str | Path,
        interface_checkpoint: str | Path,
        interface_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate with learned ReCoAlign context when the backend supports injection.

        This is deliberately not implemented as prompt serialization. A backbone
        runtime must consume learned structure tokens in its multimodal hidden
        context; otherwise comprehensive ReCoAlign evaluation is blocked.
        """

        del prompt, image, interface_checkpoint, interface_config, kwargs
        raise NotImplementedError(
            f"{self.model_id} does not expose learned-interface context injection"
        )

    @abstractmethod
    def encode_image(self, images: Sequence[str | Path]) -> np.ndarray:
        """Return one representation per image when supported by the backbone."""

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        del texts
        raise NotImplementedError(f"{self.model_id} does not expose text embeddings")

    def encode_visual_tokens(self, images: Sequence[str | Path]) -> np.ndarray:
        """Return projected visual-token representations when the backbone exposes them."""

        del images
        raise NotImplementedError(f"{self.model_id} does not expose projected visual tokens")

    def encode_visual_hidden(self, images: Sequence[str | Path]) -> np.ndarray:
        """Return language-side visual hidden states when the backbone exposes them."""

        del images
        raise NotImplementedError(f"{self.model_id} does not expose visual hidden states")

    def reconstruct_graph(self, record: SceneRecord) -> Any:
        """Optionally reconstruct a graph from an image for the diagnosis protocol."""

        del record
        raise NotImplementedError(f"{self.model_id} does not expose graph reconstruction")

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def supports_interface_injection(self) -> bool:
        """Whether the active runtime consumes learned tokens in hidden context."""

        return False

    def ensure_loaded(self) -> BaseVLM:
        return self if self.loaded else self.load()

    def count_tokens(self, text: str) -> int:
        return lexical_token_count(text)

    def prepare_input(
        self,
        record: SceneRecord,
        condition: str,
        *,
        setting: str = InputSetting.NATURAL.value,
        token_match_tolerance: int = 1,
    ) -> PreparedInput:
        question = self.prompt_protocol.render_question(record.question, record.choices)
        evidence = controlled_evidence_view(
            record,
            condition,
            setting=setting,
            count_tokens=lambda text: self.count_tokens(
                self.prompt_protocol.separator.join((text, question))
            ),
            tolerance=token_match_tolerance,
        )
        prompt = self.prompt_protocol.render(evidence.text, record.question, record.choices)
        return PreparedInput(
            prompt=prompt,
            image=condition_image(record, condition),
            condition=str(condition),
            setting=str(setting),
            input_tokens=self.count_tokens(prompt),
            evidence_tokens=self.count_tokens(evidence.text),
            semantic_units=evidence.semantic_units,
            relation_count=evidence.relation_count,
            semantic_facts_sha256=evidence.semantic_facts_sha256,
            serialization=evidence.serialization,
            padding_units=evidence.padding_units,
            token_match_delta=int(evidence.token_match_delta or 0),
            prompt_protocol_id=self.prompt_protocol.protocol_id,
            prompt_protocol_sha256=self.prompt_protocol.sha256,
        )

    def reason(
        self,
        record: SceneRecord,
        condition: str,
        *,
        prepared_input: PreparedInput | None = None,
        setting: str = InputSetting.NATURAL.value,
        token_match_tolerance: int = 1,
    ) -> str:
        prepared = prepared_input or self.prepare_input(
            record,
            condition,
            setting=setting,
            token_match_tolerance=token_match_tolerance,
        )
        cache_payload = {
            "model": self.model_id,
            "prompt": hashlib.sha256(prepared.prompt.encode("utf-8")).hexdigest(),
            "image": record.metadata.get("image_sha256", str(prepared.image)),
            "prompt_protocol": prepared.prompt_protocol_sha256,
            "generation": self.prompt_protocol.generation_config(),
        }
        key = self.inference_cache.key(cache_payload)
        cached = self.inference_cache.get(key)
        if cached is not None:
            return cached
        prediction = self.generate(
            prepared.prompt,
            image=prepared.image,
            **self.prompt_protocol.generation_config(),
        )
        self.inference_cache.put(key, prediction, cache_payload)
        return prediction

    def reason_batch(
        self,
        requests: Sequence[tuple[SceneRecord, str, PreparedInput]],
    ) -> list[str]:
        return [
            self.reason(record, condition, prepared_input=prepared)
            for record, condition, prepared in requests
        ]

    def evaluate_detailed(self, prediction: str, record: SceneRecord) -> EvaluationResult:
        return self.answer_evaluator.evaluate(
            prediction,
            record.answer,
            choices=record.choices,
            allow_reasoning_trace=True,
        )

    def evaluate(self, prediction: str, record: SceneRecord) -> bool:
        return self.evaluate_detailed(prediction, record).correct

    def provenance(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "loaded": self.loaded,
            "batch_size": self.batch_size,
            "prompt": self.prompt_protocol.to_dict(),
            "answer_evaluator": self.answer_evaluator.version,
            "cache_enabled": self.inference_cache.enabled,
        }


class ReferenceVLM(BaseVLM):
    """Deterministic non-neural sanity baseline; never claim-bearing evidence."""

    model_id = "reference"

    def __init__(
        self,
        *,
        seed: int = 7,
        condition_accuracy: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.seed = seed
        self.condition_accuracy = {
            Condition.IMAGE: 0.65,
            Condition.OBJECT_LIST: 0.55,
            Condition.CAPTION: 0.70,
            Condition.SCENE_GRAPH: 0.85,
            Condition.PARTIAL_GRAPH: 0.63,
            Condition.CORRUPTED_GRAPH: 0.40,
            **(condition_accuracy or {}),
        }

    def load(self) -> ReferenceVLM:
        self._loaded = True
        return self

    def generate(self, prompt: str, *, image: str | Path | None = None, **kwargs: Any) -> str:
        del image, kwargs
        for choice in ("left", "right", "above", "below", "front", "behind"):
            if choice in prompt.lower():
                return choice
        return "unknown"

    def encode_image(self, images: Sequence[str | Path]) -> np.ndarray:
        return np.stack([self._vector(str(image)) for image in images])

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vector(text) for text in texts])

    def reason(
        self,
        record: SceneRecord,
        condition: str,
        *,
        prepared_input: PreparedInput | None = None,
        setting: str = InputSetting.NATURAL.value,
        token_match_tolerance: int = 1,
    ) -> str:
        del prepared_input, setting, token_match_tolerance
        fallback = (
            Condition.CAPTION
            if condition == Condition.TEXT_UNORDERED
            else Condition.SCENE_GRAPH
            if condition
            in {
                Condition.GRAPH_ORDER_SHUFFLED,
                Condition.GRAPH_SERIALIZATION_TRIPLES,
                Condition.GRAPH_SERIALIZATION_JSON,
            }
            else condition
        )
        probability = float(
            self.condition_accuracy.get(condition, self.condition_accuracy.get(fallback, 0.5))
        )
        digest = hashlib.sha256(f"{self.seed}:{record.scene_id}:{condition}".encode()).digest()
        score = int.from_bytes(digest[:8], "big") / float(2**64)
        if score < probability:
            return record.answer
        wrong = [choice for choice in record.choices if choice != record.answer]
        return wrong[int.from_bytes(digest[8:12], "big") % len(wrong)]

    def provenance(self) -> dict[str, Any]:
        return {
            **super().provenance(),
            "seed": self.seed,
            "evidence_role": "infrastructure_validation",
        }

    @staticmethod
    def _vector(value: str, dimension: int = 16) -> np.ndarray:
        digest = hashlib.sha256(value.encode()).digest()
        raw = np.frombuffer((digest * ((dimension // len(digest)) + 1))[:dimension], dtype=np.uint8)
        vector = raw.astype(np.float32) / 255.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector


__all__ = ["BaseVLM", "PreparedInput", "ReferenceVLM"]
