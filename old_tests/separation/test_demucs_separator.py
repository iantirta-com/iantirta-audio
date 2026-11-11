import pytest
import torch

from iantirta.audio.separation.base import SeparationConfig, SeparationResult, Separator


def test_demucs_separator_initializes():
    config = SeparationConfig(
        model_name="mdx_extra_q",
        device="cpu",
    )

    separator = Separator(config=config)

    assert separator.model is not None
    assert separator.samplerate > 0
    assert separator.audio_channels > 0


def test_demucs_separator_forward():
    config = SeparationConfig(
        model_name="mdx_extra_q",
        device="cpu",
    )

    separator = Separator(config=config)

    separator.separate("test.mp3")
