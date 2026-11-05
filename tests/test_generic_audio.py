
import pytest
from iantirta.audio._audio import slice_np


def test_slice_np(make_audio):
    sr = 16000
    channels = 1
    dummy_audio = make_audio(sr=sr, channels=channels)

    achunk = slice_np(dummy_audio, 1.0, 2.0, sr=sr)
    C, T = achunk.shape
    assert C == channels
    assert T == (2.0 - 1.0) * sr


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([
        '-v', '-s',
        '--log-cli-level=DEBUG',
        __file__
    ]))
