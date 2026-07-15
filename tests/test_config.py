from copy import deepcopy

import pytest

from recoalign.config import ConfigError, config_digest, validate_config


def test_config_digest_is_order_independent(research_config) -> None:
    reordered = {key: research_config[key] for key in reversed(research_config)}
    assert config_digest(research_config) == config_digest(reordered)


def test_missing_manifest_is_rejected(research_config) -> None:
    config = deepcopy(research_config)
    del config["data"]["manifest"]
    with pytest.raises(ConfigError, match="data.manifest"):
        validate_config(config)


def test_duplicate_recall_cutoffs_are_rejected(research_config) -> None:
    config = deepcopy(research_config)
    config["evaluation"]["recall_at"] = [1, 1]
    with pytest.raises(ConfigError, match="duplicates"):
        validate_config(config)
