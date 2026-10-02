# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    ModeKind : TypeAlias = Literal[
        "constant",
        "edge",
        "linear_ramp",
        "maximum",
        "mean",
        "median",
        "minimum",
        "reflect",
        "symmetric",
        "wrap",
        "empty",
    ]
    PadMode = ModeKind | Callable[..., Any]


__all__ = [
    "rms", 
    "spectrogram",
]

def rms(
    *,
    y: np.ndarray | None = None,
    S: np.ndarray | None = None,
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    pad_mode: PadMode = "constant",
    dtype: DTypeLike = np.float32,
) -> np.ndarray:
    """Compute root-mean-square (RMS) value for each frame.

    Input can be given either as a time series ``y`` or as a
    spectrogram magnitude ``S``.
    Computing the RMS value from audio samples is faster as it doesn't require
    a STFT calculation. However, using a spectrogram will give a more accurate
    representation of energy over time because its frames can be windowed,
    thus prefer using ``S`` if it's already available.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        (optional) audio time series. Required if ``S`` is not input.
        Multi-channel is supported.
    S : np.ndarray [shape=(..., d, t)] or None
        (optional) spectrogram magnitude. Required if ``y`` is not input.
    frame_length : int > 0 [scalar]
        length of analysis frame (in samples) for energy calculation
    hop_length : int > 0 [scalar]
        hop length for STFT. See `librosa.stft` for details.
    center : bool
        If `True` and operating on time-domain input (``y``), pad the signal
        by ``frame_length//2`` on either side.
        If operating on spectrogram input, this has no effect.
    pad_mode : str
        Padding mode for centered analysis.  See `numpy.pad` for valid
        values.
    dtype : np.dtype, optional
        Data type of the output array.  Defaults to float32.

    Returns
    -------
    rms : np.ndarray [shape=(..., 1, t)]
        RMS value for each frame

    Examples
    --------
    >>> y, sr = librosa.loadx('trumpet')
    >>> librosa.feature.rms(y=y)
    array([[1.248e-01, 1.259e-01, ..., 1.845e-05, 1.796e-05]],
          dtype=float32)

    Or from spectrogram input

    >>> S, phase = librosa.magphase(librosa.stft(y))
    >>> rms = librosa.feature.rms(S=S)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> times = librosa.times_like(rms)
    >>> ax[0].semilogy(times, rms[0], label='RMS Energy')
    >>> ax[0].set(xticks=[])
    >>> ax[0].legend()
    >>> ax[0].label_outer()
    >>> librosa.display.specshow(S, vscale='dBFS',
    ...                          y_axis='log', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='log Power spectrogram')

    Use a STFT window of constant ones and no frame centering to get consistent
    results with the RMS computed from the audio samples ``y``

    >>> S = librosa.magphase(librosa.stft(y, window=np.ones, center=False))[0]
    >>> librosa.feature.rms(S=S)
    >>> plt.show()
    """
    # Require numba
    from iantirta.audio import signal

    if y is not None:
        # util.valid_audio(y)

        if center:
            padding = [(0, 0) for _ in range(y.ndim)]
            padding[-1] = (int(frame_length // 2), int(frame_length // 2))
            y = np.pad(y, padding, mode=pad_mode)

        x = signal.frame(y, frame_length=frame_length, hop_length=hop_length)

        # Calculate power
        power = np.mean(signal.abs2(x, dtype=dtype), axis=-2, keepdims=True)
    elif S is not None:
        # Check the frame length
        if S.shape[-2] != frame_length // 2 + 1:
            raise ValueError(
                f"Since S.shape[-2] is {S.shape[-2]}, "
                f"frame_length is expected to be {S.shape[-2] * 2 - 2} or {S.shape[-2] * 2 - 1}; "
                f"found {frame_length}"
            )

        # power spectrogram
        x = signal.abs2(S, dtype=dtype)

        # Adjust the DC and sr/2 component
        x[..., 0, :] *= 0.5
        if frame_length % 2 == 0:
            x[..., -1, :] *= 0.5

        # Calculate power
        power = 2 * np.sum(x, axis=-2, keepdims=True) / frame_length**2
    else:
        raise ValueError("Either `y` or `S` must be input.")

    rms_result: np.ndarray = np.sqrt(power)
    return rms_result


# Note: This method processes a single waveform. For batch processing, use spectrogram_batch().
def spectrogram(
    waveform: np.ndarray,
    window: np.ndarray,
    frame_length: int,
    hop_length: int,
    fft_length: int | None = None,
    power: float | None = 1.0,
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
    dither: float = 0.0,
    preemphasis: float | None = None,
    mel_filters: np.ndarray | None = None,
    mel_floor: float = 1e-10,
    log_mel: str | None = None,
    reference: float = 1.0,
    min_value: float = 1e-10,
    db_range: float | None = None,
    remove_dc_offset: bool = False,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Calculates a spectrogram over one waveform using the Short-Time Fourier Transform.

    This function can create the following kinds of spectrograms:

      - amplitude spectrogram (`power = 1.0`)
      - power spectrogram (`power = 2.0`)
      - complex-valued spectrogram (`power = None`)
      - log spectrogram (use `log_mel` argument)
      - mel spectrogram (provide `mel_filters`)
      - log-mel spectrogram (provide `mel_filters` and `log_mel`)

    How this works:

      1. The input waveform is split into frames of size `frame_length` that are partially overlapping by `frame_length
         - hop_length` samples.
      2. Each frame is multiplied by the window and placed into a buffer of size `fft_length`.
      3. The DFT is taken of each windowed frame.
      4. The results are stacked into a spectrogram.

    We make a distinction between the following "blocks" of sample data, each of which may have a different lengths:

      - The analysis frame. This is the size of the time slices that the input waveform is split into.
      - The window. Each analysis frame is multiplied by the window to avoid spectral leakage.
      - The FFT input buffer. The length of this determines how many frequency bins are in the spectrogram.

    In this implementation, the window is assumed to be zero-padded to have the same size as the analysis frame. A
    padded window can be obtained from `window_function()`. The FFT input buffer may be larger than the analysis frame,
    typically the next power of two.

    Note: This function is not optimized for speed yet. It should be mostly compatible with `librosa.stft` and
    `torchaudio.functional.transforms.Spectrogram`, although it is more flexible due to the different ways spectrograms
    can be constructed.

    Args:
        waveform (`np.ndarray` of shape `(length,)`):
            The input waveform. This must be a single real-valued, mono waveform.
        window (`np.ndarray` of shape `(frame_length,)`):
            The windowing function to apply, including zero-padding if necessary. The actual window length may be
            shorter than `frame_length`, but we're assuming the array has already been zero-padded.
        frame_length (`int`):
            The length of the analysis frames in samples. With librosa this is always equal to `fft_length` but we also
            allow smaller sizes.
        hop_length (`int`):
            The stride between successive analysis frames in samples.
        fft_length (`int`, *optional*):
            The size of the FFT buffer in samples. This determines how many frequency bins the spectrogram will have.
            For optimal speed, this should be a power of two. If `None`, uses `frame_length`.
        power (`float`, *optional*, defaults to 1.0):
            If 1.0, returns the amplitude spectrogram. If 2.0, returns the power spectrogram. If `None`, returns
            complex numbers.
        center (`bool`, *optional*, defaults to `True`):
            Whether to pad the waveform so that frame `t` is centered around time `t * hop_length`. If `False`, frame
            `t` will start at time `t * hop_length`.
        pad_mode (`str`, *optional*, defaults to `"reflect"`):
            Padding mode used when `center` is `True`. Possible values are: `"constant"` (pad with zeros), `"edge"`
            (pad with edge values), `"reflect"` (pads with mirrored values).
        onesided (`bool`, *optional*, defaults to `True`):
            If True, only computes the positive frequencies and returns a spectrogram containing `fft_length // 2 + 1`
            frequency bins. If False, also computes the negative frequencies and returns `fft_length` frequency bins.
        dither (`float`, *optional*, defaults to 0.0):
            Adds dithering. In other words, adds a small Gaussian noise to each frame.
            E.g. use 4.0 to add dithering with a normal distribution centered
            around 0.0 with standard deviation 4.0, 0.0 means no dithering.
            Dithering has similar effect as `mel_floor`. It reduces the high log_mel_fbank
            values for signals with hard-zero sections, when VAD cutoff is present in the signal.
        preemphasis (`float`, *optional*)
            Coefficient for a low-pass filter that applies pre-emphasis before the DFT.
        mel_filters (`np.ndarray` of shape `(num_freq_bins, num_mel_filters)`, *optional*):
            The mel filter bank. If supplied, applies a this filter bank to create a mel spectrogram.
        mel_floor (`float`, *optional*, defaults to 1e-10):
            Minimum value of mel frequency banks.
        log_mel (`str`, *optional*):
            How to convert the spectrogram to log scale. Possible options are: `None` (don't convert), `"log"` (take
            the natural logarithm) `"log10"` (take the base-10 logarithm), `"dB"` (convert to decibels). Can only be
            used when `power` is not `None`.
        reference (`float`, *optional*, defaults to 1.0):
            Sets the input spectrogram value that corresponds to 0 dB. For example, use `np.max(spectrogram)` to set
            the loudest part to 0 dB. Must be greater than zero.
        min_value (`float`, *optional*, defaults to `1e-10`):
            The spectrogram will be clipped to this minimum value before conversion to decibels, to avoid taking
            `log(0)`. For a power spectrogram, the default of `1e-10` corresponds to a minimum of -100 dB. For an
            amplitude spectrogram, the value `1e-5` corresponds to -100 dB. Must be greater than zero.
        db_range (`float`, *optional*):
            Sets the maximum dynamic range in decibels. For example, if `db_range = 80`, the difference between the
            peak value and the smallest value will never be more than 80 dB. Must be greater than zero.
        remove_dc_offset (`bool`, *optional*):
            Subtract mean from waveform on each frame, applied before pre-emphasis. This should be set to `true` in
            order to get the same results as `torchaudio.compliance.kaldi.fbank` when computing mel filters.
        dtype (`np.dtype`, *optional*, defaults to `np.float32`):
            Data type of the spectrogram tensor. If `power` is None, this argument is ignored and the dtype will be
            `np.complex64`.

    Returns:
        `nd.array` containing a spectrogram of shape `(num_frequency_bins, length)` for a regular spectrogram or shape
        `(num_mel_filters, length)` for a mel spectrogram.
    """
    window_length = len(window)

    if fft_length is None:
        fft_length = frame_length

    if frame_length > fft_length:
        raise ValueError(f"frame_length ({frame_length}) may not be larger than fft_length ({fft_length})")

    if window_length != frame_length:
        raise ValueError(f"Length of the window ({window_length}) must equal frame_length ({frame_length})")

    if hop_length <= 0:
        raise ValueError("hop_length must be greater than zero")

    if waveform.ndim != 1:
        raise ValueError(f"Input waveform must have only one dimension, shape is {waveform.shape}")

    if np.iscomplexobj(waveform):
        raise ValueError("Complex-valued input waveforms are not currently supported")

    if power is None and mel_filters is not None:
        raise ValueError(
            "You have provided `mel_filters` but `power` is `None`. Mel spectrogram computation is not yet supported for complex-valued spectrogram."
            "Specify `power` to fix this issue."
        )

    # center pad the waveform
    if center:
        padding = [(int(frame_length // 2), int(frame_length // 2))]
        waveform = np.pad(waveform, padding, mode=pad_mode)

    # promote to float64, since np.fft uses float64 internally
    waveform = waveform.astype(np.float64)
    window = window.astype(np.float64)

    # split waveform into frames of frame_length size
    num_frames = int(1 + np.floor((waveform.size - frame_length) / hop_length))

    num_frequency_bins = (fft_length // 2) + 1 if onesided else fft_length
    spectrogram = np.empty((num_frames, num_frequency_bins), dtype=np.complex64)

    # rfft is faster than fft
    fft_func = np.fft.rfft if onesided else np.fft.fft
    buffer = np.zeros(fft_length)

    timestep = 0
    for frame_idx in range(num_frames):
        buffer[:frame_length] = waveform[timestep : timestep + frame_length]

        if dither != 0.0:
            buffer[:frame_length] += dither * np.random.randn(frame_length)

        if remove_dc_offset:
            buffer[:frame_length] = buffer[:frame_length] - buffer[:frame_length].mean()

        if preemphasis is not None:
            buffer[1:frame_length] -= preemphasis * buffer[: frame_length - 1]
            buffer[0] *= 1 - preemphasis

        buffer[:frame_length] *= window

        spectrogram[frame_idx] = fft_func(buffer)
        timestep += hop_length

    # note: ** is much faster than np.power
    if power is not None:
        spectrogram = np.abs(spectrogram, dtype=np.float64) ** power

    spectrogram = spectrogram.T

    if mel_filters is not None:
        spectrogram = np.maximum(mel_floor, np.dot(mel_filters.T, spectrogram))

    if power is not None and log_mel is not None:
        if log_mel == "log":
            spectrogram = np.log(spectrogram)
        elif log_mel == "log10":
            spectrogram = np.log10(spectrogram)
        elif log_mel == "dB":
            from iantirta.audio.spectogram import amplitude_to_db, power_to_db

            if power == 1.0:
                spectrogram = amplitude_to_db(spectrogram, reference, min_value, db_range)
            elif power == 2.0:
                spectrogram = power_to_db(spectrogram, reference, min_value, db_range)
            else:
                raise ValueError(f"Cannot use log_mel option '{log_mel}' with power {power}")
        else:
            raise ValueError(f"Unknown log_mel option: {log_mel}")

        spectrogram = np.asarray(spectrogram, dtype)

    return spectrogram
