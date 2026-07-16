"""Auditable caption-content multiset policies for compositional benchmarks."""

from __future__ import annotations

from collections import Counter

WHITESPACE_TOKEN_MULTISET = "casefolded_whitespace_tokens_v1"
WINOGROUND_CONTENT_MULTISET = "casefolded_alphanumeric_character_multiset_v1"


def caption_multiset_matches(first: str, second: str, *, method: str) -> bool:
    """Compare captions with an explicit, versioned conservation policy."""
    return _caption_multiset(first, method=method) == _caption_multiset(second, method=method)


def _caption_multiset(text: str, *, method: str) -> Counter[str]:
    if method == WHITESPACE_TOKEN_MULTISET:
        return Counter(text.casefold().split())
    if method == WINOGROUND_CONTENT_MULTISET:
        return Counter(character for character in text.casefold() if character.isalnum())
    raise ValueError(f"unsupported caption multiset method: {method!r}")
