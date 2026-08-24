"""Ground-truth-blind forced-choice likelihood scoring for PIVOT_EXP_A3."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from recoalign.models.vlm.base import PreparedInput

from .answer_contract import PRIMARY_COMPLETION_PREFIX, REGISTERED_CHOICE_IDS


class ChoiceLikelihoodBackend(Protocol):
    def score_choice_log_likelihoods(
        self, prepared_input: PreparedInput, choice_ids: Sequence[str]
    ) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class ChoiceScoreResult:
    predicted_choice_id: str
    log_likelihoods: dict[str, float]
    choice_ids: tuple[str, ...]
    tied_maximum: bool
    valid_measurement: bool = True
    method: str = "conditional_log_likelihood_single_token_argmax_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_choice_id": self.predicted_choice_id,
            "log_likelihoods": dict(self.log_likelihoods),
            "choice_ids": list(self.choice_ids),
            "tied_maximum": self.tied_maximum,
            "valid_measurement": self.valid_measurement,
            "method": self.method,
        }


def score_choices(
    prepared_input: PreparedInput,
    choice_ids: Sequence[str],
    *,
    backend: ChoiceLikelihoodBackend,
) -> ChoiceScoreResult:
    """Select the registered option with maximum conditional log likelihood.

    Ground truth is intentionally absent from the function signature. A finite score for exactly
    four registered IDs always yields a prediction; deterministic ties use registered ID order.
    """

    registered = tuple(str(value) for value in choice_ids)
    if registered != REGISTERED_CHOICE_IDS:
        raise ValueError(f"choice IDs must be exactly {REGISTERED_CHOICE_IDS}, got {registered}")
    raw_scores = backend.score_choice_log_likelihoods(prepared_input, registered)
    if set(raw_scores) != set(registered):
        raise ValueError("likelihood backend must return exactly one score per registered choice")
    scores = {choice: float(raw_scores[choice]) for choice in registered}
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("choice log likelihoods must all be finite")
    maximum = max(scores.values())
    winners = [choice for choice in registered if scores[choice] == maximum]
    return ChoiceScoreResult(
        predicted_choice_id=winners[0],
        log_likelihoods=scores,
        choice_ids=registered,
        tied_maximum=len(winners) > 1,
    )


def validate_tokenizer_contract(
    tokenizer: Any,
    choice_ids: Sequence[str] = REGISTERED_CHOICE_IDS,
    *,
    completion_prefix: str = PRIMARY_COMPLETION_PREFIX,
) -> dict[str, Any]:
    """Verify the exact no-leading-space single-token scoring boundary."""

    registered = tuple(str(value) for value in choice_ids)
    prefix_ids = _token_ids(tokenizer, completion_prefix)
    rows = []
    for choice in registered:
        standalone = _token_ids(tokenizer, choice)
        leading_space = _token_ids(tokenizer, f" {choice}")
        combined = _token_ids(tokenizer, completion_prefix + choice)
        suffix = combined[len(prefix_ids) :] if combined[: len(prefix_ids)] == prefix_ids else []
        passed = len(standalone) == 1 and suffix == standalone
        rows.append(
            {
                "choice_id": choice,
                "standalone_token_ids": standalone,
                "leading_space_token_ids": leading_space,
                "prefix_context_token_ids": suffix,
                "single_token": len(standalone) == 1,
                "prefix_boundary_stable": suffix == standalone,
                "leading_space_sensitive": leading_space != standalone,
                "primary_method_uses_leading_space": False,
                "passed": passed,
            }
        )
    token_ids = [row["standalone_token_ids"][0] for row in rows if row["standalone_token_ids"]]
    return {
        "choice_ids": list(registered),
        "completion_prefix": completion_prefix,
        "completion_prefix_has_trailing_space": completion_prefix.endswith(" "),
        "leading_space_policy": (
            "Primary likelihood candidates are appended directly after ASCII '='. Leading-space "
            "encodings are measured and excluded, never substituted after inference."
        ),
        "unique_token_ids": len(token_ids) == len(set(token_ids)) == len(registered),
        "passed": all(bool(row["passed"]) for row in rows)
        and len(token_ids) == len(set(token_ids)) == len(registered),
        "options": rows,
    }


class LlavaConditionalLikelihoodBackend:
    """Adapter over the frozen LLaVA runtime; loading occurs only when scoring is called."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def score_choice_log_likelihoods(
        self, prepared_input: PreparedInput, choice_ids: Sequence[str]
    ) -> Mapping[str, float]:
        self.model.ensure_loaded()
        backend = self.model._require_backend()  # noqa: SLF001 - frozen runtime adapter boundary
        if hasattr(backend, "score_choice_log_likelihoods"):
            return backend.score_choice_log_likelihoods(prepared_input, choice_ids)
        tokenizer_report = validate_tokenizer_contract(backend.tokenizer, choice_ids)
        if not tokenizer_report["passed"]:
            raise ValueError("frozen tokenizer contract failed at inference boundary")
        token_ids = {
            row["choice_id"]: int(row["standalone_token_ids"][0])
            for row in tokenizer_report["options"]
        }
        logits = self._next_token_logits(backend, prepared_input)
        import torch

        log_probabilities = torch.log_softmax(logits.float(), dim=-1)
        return {
            choice: float(log_probabilities[token_id].detach().cpu())
            for choice, token_id in token_ids.items()
        }

    @staticmethod
    def _next_token_logits(backend: Any, prepared_input: PreparedInput) -> Any:
        import torch

        if prepared_input.image is None:
            raise ValueError("LLaVA forced choice requires a registered image input")
        if hasattr(backend, "_multimodal_embeddings"):
            embeddings = backend._multimodal_embeddings(  # noqa: SLF001
                prepared_input.prompt, prepared_input.image
            )
            attention = torch.ones(
                embeddings.shape[:2], dtype=torch.long, device=embeddings.device
            )
            with torch.inference_mode():
                output = backend.model(inputs_embeds=embeddings, attention_mask=attention)
            return output.logits[0, -1]

        with Image.open(Path(prepared_input.image)) as source:
            pixel_image = source.convert("RGB")
        formatted = backend._format_prompt(prepared_input.prompt)  # noqa: SLF001
        inputs = backend.processor(images=pixel_image, text=formatted, return_tensors="pt")
        device = next(backend.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = backend.model(**inputs)
        return output.logits[0, -1]


def _token_ids(tokenizer: Any, value: str) -> list[int]:
    encoded = tokenizer(value, add_special_tokens=False)
    return [int(token) for token in encoded["input_ids"]]


__all__ = [
    "ChoiceLikelihoodBackend",
    "ChoiceScoreResult",
    "LlavaConditionalLikelihoodBackend",
    "score_choices",
    "validate_tokenizer_contract",
]
