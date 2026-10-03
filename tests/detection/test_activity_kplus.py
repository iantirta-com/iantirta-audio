import numpy as np
import pytest

from kplus.pipelines.audio import detect_audio_activity as kplus_detect
from iantirta.audio.detection import detect_audio_activity as iantirta_detect


def test_detect_audio_activity_matches_kplus(audio):
    expected = kplus_detect(audio, sr=16000)
    actual = iantirta_detect(audio, sr=16000)

    assert len(actual) == len(expected)

    for old, new in zip(expected, actual):
        np.testing.assert_allclose(old.start, new.start, atol=0.02)
        np.testing.assert_allclose(old.end, new.end, atol=0.02)