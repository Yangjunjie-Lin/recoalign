from copy import deepcopy

import pytest

from recoalign.config import ConfigError, config_digest, validate_config

BASE_CONFIG = {
    "experiment": {"name": "baseline", "seed": 42, "output_dir": "outputs/baseline"},
    "model": {"framework": "open_clip", "name": "ViT-B-32", "pretrained": "test"},
    "data": {"dataset": "toy", "root": "data/toy", "split": "test"},
    "evaluation": {"recall_at": [1, 5, 10]},
    "training": {"enabled": False},
}


def test_config_digest_is_order_independent() -> None:
    reordered = {key: BASE_CONFIG[key] for key in reversed(BASE_CONFIG)}
    assert config_digest(BASE_CONFIG) == config_digest(reordered)


def test_missing_required_field_is_rejected() -> None:
    config = deepcopy(BASE_CONFIG)
    del config["data"]["split"]
    with pytest.raises(ConfigError, match="data.split"):
        validate_config(config)
