from __future__ import annotations

from dataclasses import dataclass

import pytest

from recoalign.construct_validity.runtime import configure_tokenizer_for_contract


@dataclass
class _Tokenizer:
    add_prefix_space: bool = True


def test_frozen_tokenizer_runtime_disables_implicit_prefix_space() -> None:
    tokenizer = _Tokenizer()
    assert configure_tokenizer_for_contract(tokenizer) is tokenizer
    assert tokenizer.add_prefix_space is False


def test_frozen_tokenizer_runtime_rejects_unknown_tokenizer_contract() -> None:
    with pytest.raises(TypeError, match="add_prefix_space"):
        configure_tokenizer_for_contract(object())
