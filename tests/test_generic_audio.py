
import pytest

from iantirta.audio._audio import slice_audio


def test_slice_audio_numpy(make_audio):
    """
    test to slice audio from numpy.
    """
    sr = 16000
    channels = 1
    start = 1.0
    end = 2.0
    duration = 5.0
    dummy_audio = make_audio(sr=sr, channels=channels, duration=duration)

    achunk = slice_audio(dummy_audio, start, end, sr=sr)
    C, T = achunk.shape
    assert C == channels
    assert T == (end - start) * sr


def test_slice_audio_torch(make_audio):
    """
    test to slice audio from torch.
    """
    sr = 16000
    channels = 1
    start = 1.0
    end = 2.0
    duration = 5.0
    dummy_audio = make_audio(sr=sr, channels=channels, duration=duration)

    import torch
    dummy_audio = torch.from_numpy(dummy_audio)

    achunk = slice_audio(dummy_audio, start, end, sr=sr)
    C, T = achunk.shape
    assert C == channels
    assert T == (end - start) * sr


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([
        '-v', '-s',
        '--log-cli-level=DEBUG',
        __file__
    ]))
