
from unittest.mock import patch

import numpy as np
import pytest

from iantirta.audio import load_audio, load_audio_utils
from iantirta.audio.load_audio_utils import _is_torchcodec_available


class TestLoadAudio:
    def assert_audio_valid(self, wav):
        assert isinstance(wav, np.ndarray)
        *shape, C, T = wav.shape

    def assert_compare_audio_valid(self, wav1, wav2, sr1, sr2):
        self.assert_audio_valid(wav1)
        self.assert_audio_valid(wav2)

        assert wav1.shape == wav2.shape
        assert sr1 == sr2

    def load_audio_video_and_test_valid(self, path1, path2, **kwargs):
        wav1, sr1 = load_audio(path1, **kwargs)
        wav2, sr2 = load_audio(path2, **kwargs)

        self.assert_compare_audio_valid(wav1, wav2, sr1, sr2)
        return wav1, sr1, wav2, sr2

    def test_load_audio(self, wav_path):
        wav, sr = load_audio(wav_path)

        self.assert_audio_valid(wav)

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param(
                {"backend": "torchcodec"}, 
                marks=pytest.mark.skipif(not _is_torchcodec_available(), reason="torchcodec not installed")
            ),
            pytest.param({"backend": "torchaudio"}),
            pytest.param({"backend": "librosa"}),
            pytest.param({"backend": "audioloader"})
        ]
    )
    def test_load_audio_backend(self, mp4_path, wav_path, kwargs):
        # Normal test
        backend_fn_name = f"_load_audio_{kwargs['backend']}"
        original_fn = getattr(load_audio_utils, backend_fn_name)

        with patch.object(load_audio_utils, backend_fn_name, wraps=original_fn) as fn:
            wav1, sr1, wav2, sr2 = self.load_audio_video_and_test_valid(
                mp4_path, wav_path,
                **kwargs
            )

        if kwargs['backend'] in ("torchaudio", "librosa"):
            # Video isn't supported using torchaudio or librosa
            assert fn.call_count == 1
        else:
            assert fn.call_count == 2

        # Sample rate test
        sr = 16000
        wav1, sr1, wav2, sr2 = self.load_audio_video_and_test_valid(
            mp4_path, wav_path,
            sampling_rate=sr,
            **kwargs,
        )
        assert sr == sr1 == sr2

        # Channels test
        channels = 1
        wav1, sr1, wav2, sr2 = self.load_audio_video_and_test_valid(
            mp4_path, wav_path,
            num_channels=channels,
            **kwargs,
        )

        *shape1, C1, T1 = wav1.shape
        *shape2, C2, T2 = wav2.shape

        assert channels == C1 == C2

        # Channels and Sample rate
        wav1, sr1, wav2, sr2 = self.load_audio_video_and_test_valid(
            mp4_path, wav_path,
            sampling_rate=sr,
            num_channels=channels,
            **kwargs,
        )

        assert sr == sr1 == sr2

        *shape1, C1, T1 = wav1.shape
        *shape2, C2, T2 = wav2.shape

        assert channels == C1 == C2

    def test_load_audio_torchaudio(self, wav_path):
        wav, sr = load_audio_utils._load_audio_torchaudio(wav_path)
        self.assert_audio_valid(wav)

        *shape, C, T = wav.shape

    @pytest.mark.parametrize(
        "channels",
        [
            1, 2
        ],
    )
    def test_load_audio_torchaudio_w_channels(self, wav_path, channels: int):
        wav, sr = load_audio_utils._load_audio_torchaudio(
            wav_path,
            num_channels=channels,
        )
        self.assert_audio_valid(wav)

        *shape, C, T = wav.shape

        assert C == channels


if __name__ == "__main__":
    import logging

    from iantirta.telemetry import netsvc
    netsvc.setup_logger("iantirta", default_level=logging.DEBUG)

    import sys
    sys.exit(pytest.main([
        '-v', '-s',
        __file__
    ]))
