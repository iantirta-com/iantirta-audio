# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def make_audio():
    def _make_audio(
        sr: int = 16000,
        channels: int = 1,
        duration: float = 5.0,
    ) -> np.ndarray:
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
    return base_dir_tests / 'test.wav'


@pytest.fixture
def separated_path() -> Path:
    return Path(__file__).parent.parent / "tests" / 'separated.wav'


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([
        '-v', '-s',
        '--log-cli-level=DEBUG',
        __file__
    ]))
