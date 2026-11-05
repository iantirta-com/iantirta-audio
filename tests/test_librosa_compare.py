import pytest
import librosa


def test_compare_spectrogram(make_audio):
    sr, c = 16000, 1
    dummy_audio = make_audio(sr=sr, channels=c)
    print(librosa.feature.melspectrogram)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([
        '-v', '-s',
        '--log-cli-level=DEBUG',
        __file__
    ]))
