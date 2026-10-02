# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from ..feature.extraction import PadMode


def detect_mel_spectrogram():
    log_mel_spec_t = spectrogram(
        
    ).T


def detect_and_uniform(
    fn: Callable,
    *args,
    uniform_size_ms: int = 60,
    precision_ms: int = 0.20,
    **kwargs,
) -> np.ndarray:
    raw_data = fn(*args, *kwargs)

    from iantirta.audio.signal import uniform_1d

    framesize = max(
        1,
        int(uniform_size_ms / precision_ms),
    )
    if framesize % 2 == 0:
        framesize += 1

    return uniform_1d(raw_data, framesize)


def detect_valleys():
    pass


def detect_audio_activity(
    audio: np.ndarray,
    *,
    sr: int = 16000,
    win_length: float = 64.0,
    hop_length: float = 16.0,
    num_mel_bins: int = 80,
    fmin: float = 80,
    fmax: float = 7600,
    mel_floor: float = 1e-10,
    win_function: str = "hann_window",
):
    from iantirta.audio import feature
    from iantirta.audio import spectrogram as spec

    sample_size = win_length * sr / 1000
    sample_stride = hop_length * sr / 1000

    rms = detect_and_uniform(feature.rms, y=audio, frame_length=sample_size, hop_length=sample_stride)

    n_fft = audio_utils.optimal_fft_length(sample_size)
    n_freqs = (n_fft // 2) + 1

    window = audio_utils.window_function(
        window_length=sample_size, 
        name=self.win_function, 
        periodic=True
    )

    mel_filters = spec.mel_filter_bank(
        num_frequency_bins=n_freqs,
        num_mel_filters=num_mel_bins,
        min_frequency=fmin,
        max_frequency=fmax,
        sampling_rate=sr,
        norm="slaney",
        mel_scale="slaney",
    )
    mel_spectro = detect_and_uniform(
        feature.spectrogram,
        audio,
        window=window,
        frame_length=sample_size,
        hop_length=sample_stride,
        fft_length=n_fft,
        mel_filters=mel_filters,
        mel_floor=mel_floor,
        log_mel="log10",
    )