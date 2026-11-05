
import pytest
from pathlib import Path
import numpy as np
from iantirta.audio import save_audio, load_audio


def assert_load_audio(audio_path, sr, dummy_audio):
    wav, _sr = load_audio(audio_path)

    np.testing.assert_equal(wav, dummy_audio)
    assert wav.shape == dummy_audio.shape
    assert _sr == sr


def print_file_size(audio_path, *args):
    audio_size = Path(audio_path).stat().st_size
    print(*args, 'File Size:', audio_size // 1024, 'Kb')


def test_save_audio(tmpdir, make_audio):
    sr = 16000
    dummy_audio = make_audio(sr=sr)

    wav_path = tmpdir / 'test.wav'
    save_audio(dummy_audio, wav_path, samplerate=sr, as_float=True)

    assert_load_audio(wav_path, sr, dummy_audio)


def test_save_audio_flac(tmpdir, make_audio):
    sr = 16000
    dummy_audio = make_audio(sr=sr)

    flac_path = tmpdir / 'test.flac'
    save_audio(dummy_audio, flac_path, samplerate=sr, bits_per_sample=24)
    print_file_size(flac_path, 'Flac')


def test_save_audio_size(tmpdir, dummy_audio, base_dir_tests):
    sr = 16000
    wav_path = tmpdir / 'test.wav'
    quality_wav_path = tmpdir / 'quality_test.wav'

    save_audio(
        dummy_audio,
        wav_path,
        samplerate=sr
    )
    save_audio(
        dummy_audio,
        quality_wav_path,
        samplerate=sr,
        as_float=True
    )

    print_file_size(wav_path, 'Wav 16bps')
    print_file_size(quality_wav_path, 'Wav 32bps')

    assert_load_audio(quality_wav_path, sr, dummy_audio)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([
        '-v', '-s',
        '--log-cli-level=DEBUG',
        __file__
    ]))
