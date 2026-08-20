"""Uniform exact, normalized, and multiple-choice answer evaluation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    prediction: str
    ground_truth: str
    normalized_prediction: str
    normalized_ground_truth: str
    correct: bool
    method: str
    matched_choice: str | None = None
    reasoning_trace: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "prediction": self.prediction,
            "ground_truth": self.ground_truth,
            "normalized_prediction": self.normalized_prediction,
            "normalized_ground_truth": self.normalized_ground_truth,
            "correct": self.correct,
            "evaluation_method": self.method,
            "matched_choice": self.matched_choice,
            "reasoning_trace": self.reasoning_trace,
        }


class AnswerEvaluator:
    """Deterministically score answers without model-specific parsing rules."""

    version = "answer-evaluator-v1"

    def evaluate(
        self,
        prediction: str,
        ground_truth: str,
        *,
        choices: tuple[str, ...] = (),
        allow_reasoning_trace: bool = True,
    ) -> EvaluationResult:
        raw = str(prediction)
        candidate, trace = self._extract_candidate(raw, allow_reasoning_trace)
        normalized = normalize_answer(candidate)
        expected = normalize_answer(ground_truth)
        if candidate.strip() == ground_truth.strip():
            return self._result(
                raw,
                ground_truth,
                normalized,
                expected,
                True,
                "exact",
                ground_truth,
                trace,
            )
        if normalized == expected:
            return self._result(
                raw, ground_truth, normalized, expected, True, "normalized", ground_truth, trace
            )
        matched = self._match_choice(normalized, choices)
        correct = matched is not None and normalize_answer(matched) == expected
        return self._result(
            raw,
            ground_truth,
            normalized,
            expected,
            correct,
            "multiple_choice" if matched is not None else "no_match",
            matched,
            trace,
        )

    @staticmethod
    def _extract_candidate(prediction: str, enabled: bool) -> tuple[str, str | None]:
        if not enabled:
            return prediction, None
        patterns = (
            r"(?:final\s+answer|answer)\s*[:：]\s*([^\n]+)",
            r"\boption\s+([A-Za-z0-9_+\-]+)\b",
        )
        for pattern in patterns:
            matches = list(re.finditer(pattern, prediction, flags=re.IGNORECASE))
            if matches:
                return matches[-1].group(1).strip(), prediction
        return prediction, None

    @staticmethod
    def _match_choice(normalized_prediction: str, choices: tuple[str, ...]) -> str | None:
        matches = []
        for choice in choices:
            normalized_choice = normalize_answer(choice)
            if re.search(
                rf"(?<!\w){re.escape(normalized_choice)}(?!\w)", normalized_prediction
            ):
                matches.append(choice)
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _result(
        prediction: str,
        ground_truth: str,
        normalized_prediction: str,
        normalized_ground_truth: str,
        correct: bool,
        method: str,
        matched_choice: str | None,
        trace: str | None,
    ) -> EvaluationResult:
        return EvaluationResult(
            prediction=prediction,
            ground_truth=ground_truth,
            normalized_prediction=normalized_prediction,
            normalized_ground_truth=normalized_ground_truth,
            correct=correct,
            method=method,
            matched_choice=matched_choice,
            reasoning_trace=trace,
        )


def normalize_answer(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    normalized = re.sub(r"^[\s\.,;:!?\"'`()\[\]{}]+", "", normalized)
    normalized = re.sub(r"[\s\.,;:!?\"'`()\[\]{}]+$", "", normalized)
    return re.sub(r"\s+", " ", normalized)


__all__ = ["AnswerEvaluator", "EvaluationResult", "normalize_answer"]
