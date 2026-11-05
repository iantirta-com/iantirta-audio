
import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def make_audio():
    def _make_audio(
        sr: int = 16000,
        channels: int = 1,
        duration: float = 5.0,
    ):
        samples = np.arange(
            round(sr * duration),
            dtype=np.float32,
        ) / sr
        signal = 0.5 * np.sin(2 * np.pi * 440 * samples)

        data = np.repeat(
            signal[:, None],
            channels,
            axis=1,
        ).T

        return data
    return _make_audio


@pytest.fixture
def dummy_audio(make_audio):
    return make_audio()


@pytest.fixture
def base_dir_tests():
    return Path(__file__).parent


@pytest.fixture
def mp4_path(base_dir_tests):
    return base_dir_tests / 'test.mp4'


@pytest.fixture
def wav_path(request, base_dir_tests):
    is_manual = request.node.get_closest_marker("manual") is not None
    if is_manual:
        return base_dir_tests / 'test.wav'
    else:
        return base_dir_tests / 'dummy_test.wav'


def test_dummy_audio(dummy_audio):
    pass
    

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([
        '-v', '-s',
        '--log-cli-level=DEBUG',
        __file__
    ]))
