
import pytest
from pathlib import Path
from iantirta.audio import load_audio


@pytest.mark.manual
def test_load_audio_compare(mp4_path, wav_path):
    wav, sr = load_audio(mp4_path)
    wav2, sr2 = load_audio(wav_path)

    assert wav.shape == wav2.shape
    assert sr == sr2


@pytest.mark.manual
def test_load_audio_compare_backend(mp4_path, wav_path):
    wav, sr = load_audio(mp4_path, backend='torchaudio')
    wav2, sr2 = load_audio(wav_path, backend='torchaudio')

    assert wav.shape == wav2.shape
    assert sr == sr2


@pytest.mark.manual
def test_load_audio_compare_sampling_rate(mp4_path, wav_path):
    wav, sr = load_audio(mp4_path, sampling_rate=16000)
    wav2, sr2 = load_audio(wav_path, sampling_rate=16000)

    assert wav.shape == wav2.shape
    assert sr == sr2


def test_load_audio(wav_path):
    sr = 16000
    channels = 1
    wav, _sr = load_audio(wav_path, sampling_rate=sr, num_channels=channels)

    assert sr == _sr
    C, T = wav.shape
    assert C == channels


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([
        '-v', '-s',
        '--log-cli-level=DEBUG',
        __file__
    ]))
