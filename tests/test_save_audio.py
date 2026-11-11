
import pytest
import logging

from pathlib import Path
import numpy as np

from iantirta.audio import save_audio, load_audio


def _is_lameenc_available() -> bool:
    try:
        import lameenc
        return True
    except ModuleNotFoundError:
        return False


class TestSaveAudio:
    def assert_load_compare_audio(
        self, path, wav1, sr1,
        quality: bool = False
    ):
        wav2, sr2 = load_audio(path)

        if quality:
            np.testing.assert_equal(wav1, wav2)
        else:
            np.testing.assert_almost_equal(wav1, wav2)

        assert sr1 == sr2
        assert wav1.size == wav2.size

    def calculate_audio_correlation(self, original, reconstructed):
        import scipy.signal

        corr = scipy.signal.correlate(
            original, reconstructed,
        )
        return corr

    def test_save_audio(self, tmpdir, make_audio):
        sr = 16000
        dummy_audio = make_audio(sr=sr)

        wav_path = tmpdir / 'test.wav'
        save_audio(
            dummy_audio,
            wav_path,
            samplerate=sr,
            as_float=True
        )

        self.assert_load_compare_audio(
            wav_path, dummy_audio, sr, True
        )

    def test_save_audio_flac(self, tmpdir, make_audio):
        sr = 16000
        dummy_audio = make_audio(sr=sr)

        flac_path = tmpdir / 'test.flac'
        save_audio(
            dummy_audio,
            flac_path,
            samplerate=sr,
            bits_per_sample=24
        )

        self.assert_load_compare_audio(
            flac_path, dummy_audio, sr
        )

    @pytest.mark.skipif(not _is_lameenc_available(), reason="lameenc not installed")
    def test_save_audio_mp3(self, tmpdir, make_audio):
        sr = 16000
        dummy_audio = make_audio(sr=sr)

        mp3_path = tmpdir / 'test.mp3'
        save_audio(
            dummy_audio,
            mp3_path,
            samplerate=sr,
            preset=2,
            bitrate=320
        )

        # self.assert_load_compare_audio(
        #     mp3_path, dummy_audio, sr,
        # )

    def test_save_audio_compare_filesize(self, tmpdir, make_audio, caplog):
        """
        This test should be last as it will print
        data of file size and compare percent of audio quality.
        """
        normal_wav = tmpdir / 'test.wav'
        quality_wav = tmpdir / 'qtest.wav'
        normal_flac = tmpdir / 'test.flac'
        # Quality flac is still lower
        # than true quality wav
        quality_flac = tmpdir / 'qtest.flac'
        normal_mp3 = tmpdir / 'test.mp3'
        # Quality mp3 is the same as normal unless
        # normal is set different
        quality_mp3 = tmpdir / 'qtest.mp3'

        audio_paths = [
            normal_wav, quality_wav,
            normal_flac, quality_flac,
            normal_mp3, quality_mp3
        ]

        sr = 16000
        dummy_audio = make_audio(sr=sr)

        quality_kwargs = {
            # Wav
            "as_float": True,
            # Flac (bits per sample is also used by wav,
            # but if we set it as as_float above it will
            # override this bits per sample and set it to 32).
            "bits_per_sample": 24,
            # Mp3 (default is the same as quality)
            "preset": 2,
            "bitrate": 320,
        }

        with caplog.at_level(logging.WARNING):
            print()
            for path in audio_paths:
                is_quality = True if Path(path).name.startswith('qtest') else False
                save_audio(
                    dummy_audio, path, samplerate=sr,
                    **quality_kwargs if is_quality else {}
                )

                wav2, sr2 = load_audio(path)

                assert sr2 == sr

                corr = self.calculate_audio_correlation(
                    dummy_audio, wav2
                )
    
                path = Path(path)
                size = path.stat().st_size
                is_quality_str = f"High-Q: | " if is_quality else ""
                print(
                    f"[STAT] Format: {path.suffix.ljust(5)} | "
                    f"Size: {size / 1024:6.2f} Kb | "
                    f"Quality: {np.max(corr)/100:6.8f}% "
                    + is_quality_str
                )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([
        '-v', '-s',
        '--log-cli-level=DEBUG',
        __file__
    ]))
