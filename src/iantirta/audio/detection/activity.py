# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from ..feature.extraction import PadMode


# def detect_mel_spectrogram():
#     log_mel_spec_t = spectrogram(
        
#     ).T

def _uniform_1d(
    arr: np.ndarray,
    *,
    uniform_size_ms: int = 60,
    precision_ms: int = 0.20,
) -> np.ndarray:
    from iantirta.audio.signal import uniform_1d

    framesize = max(
        1,
        int(uniform_size_ms / precision_ms),
    )
    if framesize % 2 == 0:
        framesize += 1

    return uniform_1d(arr, framesize)


def detect_and_uniform(
    fn: Callable,
    *args,
    uniform_size_ms: int = 60,
    precision_ms: int = 0.20,
    **kwargs,
) -> np.ndarray:
    raw_data = fn(*args, *kwargs)

    return _uniform_1d(
        raw_data,
        uniform_size_ms=uniform_size_ms,
        precision_ms=precision_ms
    )


def detect_log_mel_spectrogram(
    audio: np.ndarray,
    sample_size,
    sample_stride,
    *,
    sr: int = 16000,
    num_mel_bins: int = 80,
    fmin: float = 80,
    fmax: float = 7600,
    mel_floor: float = 1e-10,
    win_function: str = "hann_window",
):
    from iantirta.audio import feature
    from iantirta.audio import spectrogram as spec

    n_fft = spec.optimal_fft_length(sample_size)
    n_freqs = (n_fft // 2) + 1

    window = spec.window_function(
        window_length=sample_size, 
        name=win_function, 
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

    log_mel_spectro = feature.spectrogram(
        audio,
        window=window,
        frame_length=sample_size,
        hop_length=sample_stride,
        fft_length=n_fft,
        mel_filters=mel_filters,
        mel_floor=mel_floor,
        log_mel="log10",
    ).T

    return _uniform_1d(log_mel_spectro)

def detect_valley(
    arr: np.ndarray,
    *,
    peaks_prominence: float = 0.01
) -> np.ndarray:


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
    

    sample_size = win_length * sr / 1000
    sample_stride = hop_length * sr / 1000

    rms = detect_and_uniform(feature.rms, y=audio, frame_length=sample_size, hop_length=sample_stride)

    log_mel_spec = detect_log_mel_spectrogram(
        audio,
        sample_size,
        sample_stride,
        sr=sr,
        num_mel_bins=num_mel_bins,
        fmax=fmax,
        fmin=fmin,
        win_function=win_function,
        mel_floor=mel_floor,
    )

    floor_percentile: int = 5
    std_multiplier: float = 0.2
    rms_noise_floor = np.percentile(rms, floor_percentile)
    rms_threshold = rms_noise_floor + (np.std(rms) * std_multiplier)
    mel_noise_floor = np.percentile(log_mel_spec, floor_percentile)
    mel_threshold = mel_noise_floor + (np.std(log_mel_spec) * std_multiplier)

    rms_mask = (rms > rms_threshold)
    mel_mask = (log_mel_spec > mel_threshold)

    final_mask = rms_mask | mel_mask

    from iantirta.audio import signal
    peaks_prominence: float = 0.01
    _max = np.max(np.abs(smoothed_rms))
    norm = (smoothed_rms / _max) if _max > 0 else smoothed_rms
    inverted = -norm
    rms_valleys = signal.find_peaks_1d(
        inverted,
        prominence=peaks_prominence
    )
