
import pytest

from iantirta.audio.separation import Separator, SeparationConfig

def test_separator():
    separator = Separator()
    assert separator is not None

def test_default_config():
    config = SeparationConfig()

    assert config.model_name == "mdx_extra_q"
    assert config.shifts == 1
    assert config.overlap == 0.25
    assert config.split is True
    assert config.segment is None
    assert config.progress is False
    assert config.num_workers == 0


def test_config_from_dict():
    config = SeparationConfig(
        model_name="mdx_extra_q",
        shifts=2,
        overlap=0.5,
    )

    assert config.model_name == "mdx_extra_q"
    assert config.shifts == 2
    assert config.overlap == 0.5


def test_config_does_not_create_tuple_defaults():
    config = SeparationConfig()

    assert isinstance(config.split, bool)
    assert isinstance(config.progress, bool)
