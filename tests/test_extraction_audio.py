
import pytest

from iantirta.audio.audio_extraction import AudioExtraction


def test_audio_extraction_all(make_audio):
    sr, c = 16000, 1
    dummy_audio = make_audio(sr=sr, channels=c)

    audio_metadata = AudioExtraction(dummy_audio, sr)
    print(audio_metadata.rms)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([
        '-v', '-s',
        '--log-cli-level=DEBUG',
        __file__
    ]))
