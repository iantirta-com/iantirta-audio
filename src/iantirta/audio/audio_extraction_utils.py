from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from iantirta.audio.extraction_utils import sequence, spectrogram

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_mask_starts_ends(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Method to get starts and ends of a mask np.ndarray

    :return:
        tuple of starts, ends, representing start, end
        in frame indices
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    diffs = np.diff(np.pad(
        arr.astype(int),
        pad_width=1,
        constant_values=0
    ))
    starts = np.where(diffs == 1)[0]
    ends   = np.where(diffs == -1)[0]
    return starts, ends


def mask_merge_forward(
    arr: np.ndarray,
    target: np.ndarray,
    *,
    max_distance_ms: int = 200,
    hop_duration_ms: int,
) -> np.ndarray:
    """
    Method that is used to merge array
    into targeted array
    """
    edges = np.where(
        (arr[:-1] == True) & (arr[1:] == False)
    )[0]

    max_frame_size = max(1, round(max_distance_ms / hop_duration_ms))

    for e in edges:
        max_dist = e + max_frame_size
        after = target[target > e]

        if len(after) > 0:
            after = after[0]
            arr[e:min(after, max_dist)] = True

    return arr


def mask_uniform(
    arr: np.ndarray,
    dist_ms: int = 140,
    *,
    hop_duration_ms: int,
):
    """ Any distance `lower than or equal` the threshold ms, would be merged """
    starts, ends = get_mask_starts_ends(arr)
    merge_frame = ms2frame(dist_ms, hop_duration_ms=hop_duration_ms)

    for s, e in zip(ends[:-1], starts[1:]): # i.e. the end of one True block, the start of the next True block
        dist = e - s
        if dist <= merge_frame:
            arr[s:e] = True

    return arr


def mask_delete(
    arr: np.ndarray,
    delete_ms: int = 100,
    *,
    hop_duration_ms: int,
) -> np.ndarray:
    """ Any mask length `lower than or equal` the threshold ms, would be deleted """
    starts, ends = get_mask_starts_ends(arr)
    delete_frame = ms2frame(delete_ms, hop_duration_ms=hop_duration_ms)

    for s, e in zip(starts, ends):
        dur = e - s

        if dur <= delete_frame:
            arr[s:e] = False

    return arr


def mask_merge_lower_duration(
    arr: np.ndarray,
    merge_ms: int = 3000,
    *,
    hop_duration_ms: int,
) -> np.ndarray:
    """ Any mask frame `lower than or equal` the threshold ms, would be merged to the nearest mask """

    while True:

        starts, ends = get_mask_starts_ends(arr)
        merge_frame = ms2frame(merge_ms, hop_duration_ms=hop_duration_ms)

        if len(starts) == 0:
            break

        short_segments = []
        for i, (s, e) in enumerate(zip(starts, ends)):
            dur = e - s
            if dur <= merge_frame:
                short_segments.append((i, s, e))

        if not short_segments:
            break

        assert len(starts) > 1

        for i, s, e in short_segments:
            next_start = starts[i+1] if i < len(starts) - 1 else None
            prev_end = ends[i-1] if i > 0 else None

            prev_gap = (s - prev_end) if prev_end is not None else float("inf")
            next_gap = (next_start - e) if next_start is not None else float("inf")

            if prev_gap < next_gap:
                arr[prev_end:s] = True

            elif next_gap < prev_gap:
                arr[e:next_start] = True

            elif next_gap == prev_gap:
                next_end = ends[i+1] if i < len(ends) - 1 else None
                next_dur = (
                    (next_end - next_start)
                    if (
                        next_start is not None
                        and next_end is not None
                    )
                    else float("inf")
                )

                prev_start = starts[i-1] if i > 0 else None
                prev_dur = (
                    (prev_end - prev_start)
                    if (
                        prev_start is not None
                        and prev_end is not None
                    ) else float("inf")
                )

                if next_dur < prev_dur:
                    arr[e:next_start] = True

                else:
                    arr[prev_end:s] = True

            else:
                raise RuntimeError("Not sure why this happen")

    return arr


def split_long_block_mask(
    arr: np.ndarray,
    target_valleys: np.ndarray,
    mel_mask: np.ndarray,
    *,
    max_maskblock_ms: int = 10000,  # 10s
    min_maskblock_ms: int = 1000,  # 1s
    min_silence_ms: int = 80,  # 10ms
    hop_duration_ms: int,
) -> np.ndarray:
    """ Split Long block Mask Width """
    max_frame = ms2frame(max_maskblock_ms, hop_duration_ms=hop_duration_ms)
    min_frame = ms2frame(min_maskblock_ms, hop_duration_ms=hop_duration_ms)

    min_silence_frame = ms2frame(min_silence_ms, hop_duration_ms=hop_duration_ms)

    starts, ends = get_mask_starts_ends(arr)

    for start, end in zip(starts, ends):
        chunks = [(start, end)]

        while any((
            (ce - cs) >= max_frame
            for cs, ce in chunks
        )):

            new_chunks = []
            for cs, ce in chunks:

                if (ce - cs) >= max_frame:

                    valleys = [
                        v for v in target_valleys
                        if (cs + min_frame) <= v <= (ce - min_frame) # - gapframe?
                    ]

                    candidates = []
                    for v in valleys:

                        if not mel_mask[v]:
                            # mel mask is False, find start and end?
                            past_mask = mel_mask[cs:v]
                            after_mask = mel_mask[v:ce]
                            true_mask = np.where(past_mask == True)[0]

                            if len(true_mask) > 0:
                                mask_start = cs + true_mask[-1] + 1
                            else:
                                mask_start = cs

                            true_mask = np.where(after_mask == True)[0]

                            if len(true_mask) > 0:
                                mask_end = v + true_mask[0]
                            else:
                                mask_end = ce

                            assert mask_end >= mask_start, (
                                "Negative value for gap"
                                f"\n  maskstart: {mask_start}"
                                f"\n  maskend: {mask_end}"
                            )
                            candidates.append({
                                "start": mask_start,
                                "end": mask_end,
                                "gap": mask_end - mask_start
                            })

                    if candidates:
                        valid_candidates = [
                            c for c in candidates
                            if c["gap"] >= min_silence_frame
                        ]

                        if valid_candidates:
                            # pick the widest gap
                            best = max(
                                valid_candidates,
                                key=lambda c: c["gap"]
                            )

                            best_start, best_end = best["start"], best["end"]
                            arr[best_start:best_end] = False

                            new_chunks.extend([(cs, best_start), (best_end, ce)])
                        else:
                            new_chunks.append((cs, ce))

                else:
                    new_chunks.append((cs, ce))

            if chunks == new_chunks:
                break

            chunks = new_chunks

    return arr


def ms2frame(
    ms: int,
    *,
    hop_duration_ms: int,
) -> int:
    return max(1, round(ms / hop_duration_ms))


def uniform_1d(
    arr: np.ndarray,
    *,
    uniform_size_ms: int = 60,
    hop_duration_ms: int,
) -> np.ndarray:
    """
    Extension of ``scipy.ndimage.uniform_filter1d``
    """
    import scipy.ndimage

    framesize = ms2frame(uniform_size_ms, hop_duration_ms=hop_duration_ms)
    if framesize % 2 == 0:
        framesize += 1

    return scipy.ndimage.uniform_filter1d(arr, size=framesize)


def optimal_fft_length(window_length: int) -> int:
    """
    Finds the best FFT input size for a given `window_length`. This function takes a given window length and, if not
    already a power of two, rounds it up to the next power or two.

    The FFT algorithm works fastest when the length of the input is a power of two, which may be larger than the size
    of the window or analysis frame. For example, if the window is 400 samples, using an FFT input size of 512 samples
    is more optimal than an FFT size of 400 samples. Using a larger FFT size does not affect the detected frequencies,
    it simply gives a higher frequency resolution (i.e. the frequency bins are smaller).
    """
    return 2 ** int(np.ceil(np.log2(window_length)))

# ------------------
# Pyin Specific helper Method
# ------------------

def _cumulative_mean_normalized_difference(
    y_frames: np.ndarray,
    min_period: int,
    max_period: int,
) -> np.ndarray:
    """
    Cumulative mean normalized difference function (equation 8 in [#]_)

    .. [#] De Cheveigné, Alain, and Hideki Kawahara.
        "YIN, a fundamental frequency estimator for speech and music."
        The Journal of the Acoustical Society of America 111.4 (2002): 1917-1930.
    """
    acf_frames = autocorrelate(y_frames, max_size=max_period + 1, axis=-2)

    # Energy terms.
    yin_frames = np.square(y_frames)
    np.cumsum(yin_frames, out=yin_frames, axis=-2)

    # Difference function: d(k) = 2 * (ACF(0) - ACF(k)) - sum_{m=0}^{k-1} y(m)^2
    k = slice(1, max_period + 1)
    yin_frames[..., 0, :] = 0
    yin_frames[..., k, :] = (
        2 * (acf_frames[..., 0:1, :] - acf_frames[..., k, :])
        - yin_frames[..., : k.stop - 1, :]
    )

    # Cumulative mean normalized difference function.
    yin_numerator = yin_frames[..., min_period : max_period + 1, :]
    # broadcast this shape to have leading ones
    k_range = expand_to(np.r_[k], ndim=yin_frames.ndim, axes=-2)

    cumulative_mean = np.cumsum(yin_frames[..., k, :], axis=-2) / k_range
    yin_denominator = cumulative_mean[..., min_period - 1 : max_period, :]
    yin_frames = yin_numerator / (
        yin_denominator + tiny(yin_denominator)
    )
    return np.asarray(yin_frames)


def __check_yin_params(*, sr: float, fmax: float, fmin: float, frame_length: int):
    """
    Check the feasibility of yin/pyin parameters against
    the following conditions:

    1. 0 < fmin < fmax <= sr/2
    2. sr / fmin < frame_length - 1: At least one period of fmin needs to fit into the frame.

    Give a warning if parameters violate:
    3. sr / fmin < frame_length / 2: Ideally, at least two periods of fmin should fit into the frame.
    """
    if fmax > sr / 2:
        raise ValueError(f"fmax={fmax:.3f} cannot exceed Nyquist frequency {sr/2}")
    if fmin >= fmax:
        raise ValueError(f"fmin={fmin:.3f} must be less than fmax={fmax:.3f}")
    if fmin <= 0:
        raise ValueError(f"fmin={fmin:.3f} must be strictly positive")

    if sr / fmin >= frame_length - 1:
        fmin_feasible = sr / (frame_length - 1)
        frame_length_feasible = int(np.ceil(sr / fmin) + 1)
        raise ValueError(
            f"fmin={fmin:.3f} is too small for frame_length={frame_length} and sr={sr}. "
            f"Either increase to fmin={fmin_feasible:.3f} or frame_length={frame_length_feasible}"
        )

    if sr / fmin >= frame_length // 2:
        fmin_optimal = sr / (frame_length / 2)
        frame_length_optimal = int(np.ceil(sr / fmin) * 2 + 1)

        warnings.warn(
            f"With fmin={fmin:.3f}, sr={sr} and frame_length={frame_length}, less than two periods of fmin "
            f"fit into the frame, which can cause inaccurate pitch detection. "
            f"Consider increasing to fmin={fmin_optimal:.3f} or frame_length={frame_length_optimal}.",
            stacklevel=3,
        )


def __pyin_helper(
    yin_frames,
    parabolic_shifts,
    sr,
    thresholds,
    boltzmann_parameter,
    beta_probs,
    no_trough_prob,
    min_period,
    fmin,
    n_pitch_bins,
    n_bins_per_semitone,
):
    yin_probs = np.zeros_like(yin_frames)

    for i, yin_frame in enumerate(yin_frames.T):
        # 2. For each frame find the troughs.
        is_trough = localmin(yin_frame)

        is_trough[0] = yin_frame[0] < yin_frame[1]
        (trough_index,) = np.nonzero(is_trough)

        if len(trough_index) == 0:
            continue

        # 3. Find the troughs below each threshold.
        # these are the local minima of the frame, could get them directly without the trough index
        trough_heights = yin_frame[trough_index]
        trough_thresholds = np.less.outer(trough_heights, thresholds[1:])

        # 4. Define the prior over the troughs.
        # Smaller periods are weighted more.
        trough_positions = np.cumsum(trough_thresholds, axis=0) - 1
        n_troughs = np.count_nonzero(trough_thresholds, axis=0)

        import scipy.stats

        trough_prior = scipy.stats.boltzmann.pmf(
            trough_positions, boltzmann_parameter, n_troughs
        )

        trough_prior[~trough_thresholds] = 0

        # 5. For each threshold add probability to global minimum if no trough is below threshold,
        # else add probability to each trough below threshold biased by prior.

        probs = trough_prior.dot(beta_probs)

        global_min = np.argmin(trough_heights)
        n_thresholds_below_min = np.count_nonzero(~trough_thresholds[global_min, :])
        probs[global_min] += no_trough_prob * np.sum(
            beta_probs[:n_thresholds_below_min]
        )

        yin_probs[trough_index, i] = probs

    yin_period, frame_index = np.nonzero(yin_probs)

    # Refine peak by parabolic interpolation.
    period_candidates = min_period + yin_period
    period_candidates = period_candidates + parabolic_shifts[yin_period, frame_index]
    f0_candidates = sr / period_candidates

    # Find pitch bin corresponding to each f0 candidate.
    bin_index = 12 * n_bins_per_semitone * np.log2(f0_candidates / fmin)
    bin_index = np.clip(np.round(bin_index), 0, n_pitch_bins).astype(int)

    # Observation probabilities.
    observation_probs = np.zeros((2 * n_pitch_bins, yin_frames.shape[1]))
    observation_probs[bin_index, frame_index] = yin_probs[yin_period, frame_index]

    voiced_prob = np.clip(
        np.sum(observation_probs[:n_pitch_bins, :], axis=0, keepdims=True), 0, 1
    )
    observation_probs[n_pitch_bins:, :] = (1 - voiced_prob) / n_pitch_bins

    return observation_probs[np.newaxis], voiced_prob


# ------------------
# Helper Method
# ------------------

def frame(
    waveform: np.ndarray,
    frame_length: int,
    hop_length: int,
):
    """
    Slice a data array into (overlapping) frames.

    This implementation uses low-level stride manipulation to avoid
    making a copy of the data.  The resulting frame representation
    is a new view of the same input data.
    """
    strides = waveform.strides + (waveform.strides[-1],)

    y_shape_trimmed = list(waveform.shape)
    y_shape_trimmed[-1] -= frame_length - 1
    shape = tuple(y_shape_trimmed) + (frame_length,)

    frames = np.lib.stride_tricks.as_strided(
        waveform,
        shape=shape,
        strides=strides,
        subok=False,
        writeable=False,
    )
    x = np.moveaxis(frames, -1, -2)

    slices = [slice(None)] * x.ndim
    slices[-1] = slice(0, None, hop_length)
    x = x[tuple(slices)]
    return x


def expand_to(
    x: np.ndarray,
    *,
    ndim: int,
    axes: int | slice | Sequence[int] | Sequence[slice]
) -> np.ndarray:
    """
    Expand the dimensions of an input array with

    Parameters
    ----------
    x : np.ndarray
        The input array
    ndim : int
        The number of dimensions to expand to.  Must be at least ``x.ndim``
    axes : int or slice
        The target axis or axes to preserve from x.
        All other axes will have length 1.

    Returns
    -------
    x_exp : np.ndarray
        The expanded version of ``x``, satisfying the following:
            ``x_exp[axes] == x``
            ``x_exp.ndim == ndim``
    """
    # Force axes into a tuple
    axes_tup: tuple[int]
    try:
        axes_tup = tuple(axes)
    except TypeError:
        axes_tup = (axes,)

    if len(axes_tup) != x.ndim:
        raise ValueError(
            f"Shape mismatch between axes={axes_tup} and input x.shape={x.shape}"
        )

    if ndim < x.ndim:
        raise ValueError(
            f"Cannot expand x.shape={x.shape} to fewer dimensions ndim={ndim}"
        )

    shape: list[int] = [1] * ndim
    for i, axi in enumerate(axes_tup):
        shape[axi] = x.shape[i]

    return x.reshape(shape)


def tiny(
    x: complex | np.ndarray
) -> np.floating[Any]:
    """Compute the tiny-value corresponding to an input's data type.

    This is the smallest "usable" number representable in ``x.dtype``
    (e.g., float32).

    This is primarily useful for determining a threshold for
    numerical underflow in division or multiplication operations.

    Parameters
    ----------
    x : number (real or complex) or np.ndarray
        The array to compute the tiny-value for.
        All that matters here is ``x.dtype``

    Returns
    -------
    tiny_value : float
        The smallest positive usable number for the type of ``x``.
        If ``x`` is integer-typed, then the tiny value for ``np.float32``
        is returned instead.
    """
    # Make sure we have an array view
    x = np.asarray(x)

    # Only floating types generate a tiny
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(
        x.dtype, np.complexfloating
    ):
        dtype = x.dtype
    else:
        dtype = np.dtype(np.float32)

    tiny: np.floating[Any] = np.finfo(dtype).tiny
    return tiny

# ------------------
# API Method
# ------------------

def rms(
    waveform: np.ndarray,
    frame_length: int,
    hop_length: int,
    *,
    center: bool = True,
    pad_mode: str = "reflect",
):
    """
    Get root mean square of audio.
    """
    if hop_length <= 0:
        raise ValueError("hop_length must be greater than zero")

    if waveform.ndim != 1:
        raise ValueError(f"Input waveform must have only one dimension, shape is {waveform.shape}")

    if np.iscomplexobj(waveform):
        raise ValueError("Complex-valued input waveforms are not currently supported")

    y = waveform
    if center:
        padding = [(0, 0) for _ in range(y.ndim)]
        padding[-1] = (int(frame_length // 2), int(frame_length // 2))
        y = np.pad(y, padding, mode=pad_mode)

    y = y.astype(np.float64)

    x = frame(y, frame_length= frame_length, hop_length=hop_length)

    rms = (np.mean(x ** 2, axis=-2, keepdims=True)) ** 0.5
    return rms


def stft(
    waveform: np.ndarray,
    window: np.ndarray,
    frame_length: int,
    hop_length: int,
    fft_length: int | None = None,
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
    dither: float = 0.0,
    preemphasis: float | None = None,
    remove_dc_offset: bool = False,
):
    """
    Short-time Fourier transform (STFT).

    The STFT represents a signal in the time-frequency domain by
    computing discrete Fourier transforms (DFT) over short overlapping
    windows.
    """
    return spectrogram.spectrogram(
        waveform=waveform,
        window=window,
        frame_length=frame_length,
        hop_length=hop_length,
        fft_length=fft_length,
        power=None,
        center=center,
        pad_mode=pad_mode,
        onesided=onesided,
        dither=dither,
        preemphasis=preemphasis,
        mel_filters=None,
        remove_dc_offset=remove_dc_offset,
    )


def melspectrogram(
    waveform: np.ndarray,
    window: np.ndarray,
    frame_length: int,
    hop_length: int,
    fft_length: int,
    power: float | None = 2.0,
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
    dither: float = 0.0,
    preemphasis: float | None = None,
    mel_filters: np.ndarray | None = None,
    mel_floor: float = 1e-10,
    log_mel: Literal['log', 'log10', 'dB'] | None = None,
    reference: float = 1.0,
    min_value: float = 1e-10,
    db_range: float | None = None,
    remove_dc_offset: bool = False,
    dtype: np.dtype = np.float32,
    *,
    sampling_rate: int,
    fmin: float = 80,
    fmax: float = 7600,
    num_mel_filters: int = 80,
):
    """
    Return Log/Mel scaled Spectrogram.

      - amplitude spectrogram (`power = 1.0`)
      - power spectrogram (`power = 2.0`)
      - complex-valued spectrogram (`power = None`)
      - log spectrogram (use `log_mel` argument)
      - log-mel spectrogram (provide `log_mel`)
    """
    n_freqs = (fft_length // 2) + 1

    mel_filters = spectrogram.mel_filter_bank(
        num_frequency_bins=n_freqs,
        num_mel_filters=num_mel_filters,
        min_frequency=fmin,
        max_frequency=fmax,
        sampling_rate=sampling_rate,
        norm="slaney",
        mel_scale="slaney",
        triangularize_in_mel_space=False,
    )

    return spectrogram.spectrogram(
        waveform=waveform,
        window=window,
        frame_length=frame_length,
        hop_length=hop_length,
        fft_length=fft_length,
        power=power,
        center=center,
        pad_mode=pad_mode,
        onesided=onesided,
        dither=dither,
        preemphasis=preemphasis,
        mel_filters=mel_filters,
        mel_floor=mel_floor,
        log_mel=log_mel,
        reference=reference,
        min_value=min_value,
        db_range=db_range,
        remove_dc_offset=remove_dc_offset,
        dtype=dtype,
    )


def autocorrelate(
    waveform: np.ndarray,
    max_size: int | None = None,
    axis: int = -1
):
    """
    Bounded-lag auto-correlation

    Parameters
    ----------
    max_size : int > 0 or None
        maximum correlation lag.
        If unspecified, defaults to ``y.shape[axis]`` (unbounded)
    
    Returns
    -------
    z : np.ndarray
        truncated autocorrelation ``y*y`` along the specified axis.
        If ``max_size`` is specified, then ``z.shape[axis]`` is bounded
        to ``max_size``.
    """
    if np.iscomplexobj(waveform):
        raise ValueError("Complex-valued input waveforms are not currently supported")

    if max_size is None:
        max_size = waveform.shape[axis]

    max_size = int(min(max_size, waveform.shape[axis]))

    import scipy.fft

    n_pad = scipy.fft.next_fast_len(2 * waveform.shape[axis] - 1, real=True)

    # Compute the power spectrum along the chosen axis
    powspec = (scipy.fft.rfft(waveform, n=n_pad, axis=axis))**2

    # Convert back to time domain
    autocorr = scipy.fft.irfft(powspec, n=n_pad, axis=axis)

    subslice = [slice(None)] * autocorr.ndim
    subslice[axis] = slice(max_size)

    autocorr_slice: np.ndarray = autocorr[tuple(subslice)]
    return autocorr_slice


def pyin(
    waveform: np.ndarray,
    *,
    fmin: float,
    fmax: float,
    sr: float,
    frame_length: int,
    hop_length: int,
    n_thresholds: int = 100,
    beta_parameters: tuple[float, float] = (2, 18),
    boltzmann_parameter: float = 2,
    resolution: float = 0.1,
    max_transition_rate: float = 35.92,
    switch_prob: float = 0.01,
    no_trough_prob: float = 0.01,
    fill_na: float | None = np.nan,
    center: bool = True,
    pad_mode: str = "reflect",
    transition_min_prob: float | None = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fundamental frequency (F0) estimation using probabilistic YIN (pYIN).

    pYIN [#]_ is a modificatin of the YIN algorithm [#]_ for fundamental frequency (F0) estimation.
    In the first step of pYIN, F0 candidates and their probabilities are computed using the YIN algorithm.
    In the second step, Viterbi decoding is used to estimate the most likely F0 sequence and voicing flags.

    .. [#] Mauch, Matthias, and Simon Dixon.
        "pYIN: A fundamental frequency estimator using probabilistic threshold distributions."
        2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2014.

    .. [#] De Cheveigné, Alain, and Hideki Kawahara.
        "YIN, a fundamental frequency estimator for speech and music."
        The Journal of the Acoustical Society of America 111.4 (2002): 1917-1930.
    """
    if fmin is None or fmax is None:
        raise ValueError('both "fmin" and "fmax" must be provided')

    if waveform.ndim != 1:
        raise ValueError(f"Input waveform must have only one dimension, shape is {waveform.shape}")

    if np.iscomplexobj(waveform):
        raise ValueError("Complex-valued input waveforms are not currently supported")

    __check_yin_params(sr=sr, fmax=fmax, fmin=fmin, frame_length=frame_length)

    y = waveform
    if center:
        padding = [(0, 0) for _ in range(y.ndim)]
        padding[-1] = (int(frame_length // 2), int(frame_length // 2))
        y = np.pad(y, padding, mode=pad_mode)

    y_frames = frame(y, frame_length=frame_length, hop_length=hop_length)

    # Calculate minimum and maximum periods
    min_period = int(np.floor(sr / fmax))
    max_period = min(int(np.ceil(sr / fmin)), frame_length - 1)

    # Calculate cumulative mean normalized difference function.
    yin_frames = _cumulative_mean_normalized_difference(
        y_frames, min_period, max_period
    )

    # Parabolic interpolation.
    parabolic_shifts = _parabolic_interpolation(yin_frames)

    # Find Yin candidates and probabilities.
    # The implementation here follows the official pYIN software which
    # differs from the method described in the paper.
    # 1. Define the prior over the thresholds.
    import scipy.stats

    thresholds = np.linspace(0, 1, n_thresholds + 1)
    beta_cdf = scipy.stats.beta.cdf(thresholds, beta_parameters[0], beta_parameters[1])
    beta_probs = np.diff(beta_cdf)

    n_bins_per_semitone = int(np.ceil(1.0 / resolution))
    n_pitch_bins = int(np.floor(12 * n_bins_per_semitone * np.log2(fmax / fmin))) + 1

    def _helper(a, b):
        return __pyin_helper(
            a,
            b,
            sr,
            thresholds,
            boltzmann_parameter,
            beta_probs,
            no_trough_prob,
            min_period,
            fmin,
            n_pitch_bins,
            n_bins_per_semitone,
        )

    helper = np.vectorize(_helper, signature="(f,t),(k,t)->(1,d,t),(j,t)")
    observation_probs, voiced_prob = helper(yin_frames, parabolic_shifts)

    # Construct transition matrix.
    max_semitones_per_frame = round(max_transition_rate * 12 * hop_length / sr)
    transition_width = max_semitones_per_frame * n_bins_per_semitone + 1
    # Construct the within voicing transition probabilities
    transition = sequence.transition_local(
        n_pitch_bins, transition_width, window="triangle", wrap=False
    )

    # Include across voicing transition probabilities
    t_switch = sequence.transition_loop(2, 1 - switch_prob)
    transition = np.kron(t_switch, transition)

    p_init = np.ones(2 * n_pitch_bins) / (2 * n_pitch_bins)

    states = sequence.viterbi(observation_probs, transition, p_init=p_init,
                              transition_min_prob=transition_min_prob)

    # Find f0 corresponding to each decoded pitch bin.
    freqs = fmin * 2 ** (np.arange(n_pitch_bins) / (12 * n_bins_per_semitone))
    f0 = freqs[states % n_pitch_bins]
    voiced_flag = states < n_pitch_bins

    if fill_na is not None:
        f0[~voiced_flag] = fill_na

    return f0[..., 0, :], voiced_flag[..., 0, :], voiced_prob[..., 0, :]


# ------------------
# Numba Implementation
# ------------------

try:
    import numba

except ImportError as e:
    warnings.warn(f"Importing numba error, will not use optimization, Error: {e}")

    def _parabolic_interpolation(x: np.ndarray, *, axis: int = -2) -> np.ndarray:
        """
        Piecewise parabolic interpolation for yin and pyin.

        Parameters
        ----------
        x : np.ndarray
            array to interpolate
        axis : int
            axis along which to interpolate

        Returns
        -------
        parabolic_shifts : np.ndarray [shape=x.shape]
            position of the parabola optima (relative to bin indices)

            Note: the shift at bin `n` is determined as 0 if the estimated
            optimum is outside the range `[n-1, n+1]`.
        """
        # Rotate the target axis to the end
        xi = x.swapaxes(-1, axis)

        out = np.zeros_like(xi)

        x_prev = xi[..., :-2]
        x_curr = xi[..., 1:-1]
        x_next = xi[..., 2:]

        a = x_next + x_prev - 2 * x_curr
        b = (x_next - x_prev) / 2

        valid_mask = (np.abs(b) < np.abs(a)) & (a != 0)

        np.divide(-b, a, out=out[..., 1:-1], where=valid_mask)

        return out.swapaxes(-1, axis)


    def localmin(x: np.ndarray, *, axis: int = 0) -> NDArray[np.bool]:
        """Find local minima in an array

        An element ``x[i]`` is considered a local minimum if the following
        conditions are met:

        - ``x[i] < x[i-1]``
        - ``x[i] <= x[i+1]``

        Note that the first condition is strict, and that the first element
        ``x[0]`` will never be considered as a local minimum.

        Parameters
        ----------
        x : np.ndarray [shape=(d1,d2,...)]
            input vector or array
        axis : int
            axis along which to compute local minimality

        Returns
        -------
        m : np.ndarray [shape=x.shape, dtype=bool]
            indicator array of local minimality along ``axis``
        """
        # Rotate the target axis to the end
        xi = x.swapaxes(-1, axis)

        out = np.zeros(xi.shape, dtype=bool)

        x_prev = xi[..., :-2]
        x_curr = xi[..., 1:-1]
        x_next = xi[..., 2:]

        valid_mask = (x_curr < x_prev) & (x_curr <= x_next)

        out[..., 1:-1] = valid_mask
        out[..., -1] = xi[..., -1] < xi[..., -2]

        return out.swapaxes(-1, axis)

else:

    @numba.stencil  # type: ignore
    def _pi_stencil(x: np.ndarray) -> np.ndarray:
        """Stencil to compute local parabolic interpolation"""
        a = x[1] + x[-1] - 2 * x[0]
        b = (x[1] - x[-1]) / 2

        if np.abs(b) >= np.abs(a):
            # If this happens, we'll shift by more than 1 bin
            # Suppressing types because mypy has no idea about stencils
            return 0  # type: ignore

        return -b / a  # type: ignore


    @numba.guvectorize(
        "(n)->(n)",
        cache=True,
        nopython=True,
    )  # type: ignore
    def _pi_wrapper(x: np.ndarray, y: np.ndarray) -> None:  # pragma: no cover
        """Vectorized wrapper for the parabolic interpolation stencil"""
        y[:] = _pi_stencil(x)


    def _parabolic_interpolation(x: np.ndarray, *, axis: int = -2) -> np.ndarray:
        """
        Piecewise parabolic interpolation for yin and pyin.

        Parameters
        ----------
        x : np.ndarray
            array to interpolate
        axis : int
            axis along which to interpolate

        Returns
        -------
        parabolic_shifts : np.ndarray [shape=x.shape]
            position of the parabola optima (relative to bin indices)

            Note: the shift at bin `n` is determined as 0 if the estimated
            optimum is outside the range `[n-1, n+1]`.
        """
        # Rotate the target axis to the end
        xi = x.swapaxes(-1, axis)

        # Allocate the output array and rotate target axis
        shifts = np.empty_like(x)
        shiftsi = shifts.swapaxes(-1, axis)

        # Call the vectorized stencil
        _pi_wrapper(xi, shiftsi)

        # Handle the edge condition not covered by the stencil
        shiftsi[..., -1] = 0
        shiftsi[..., 0] = 0

        return shifts


    @numba.stencil
    def _localmin_sten(x):  # pragma: no cover
        """Numba stencil for local minima computation"""
        return (x[0] < x[-1]) & (x[0] <= x[1])


    @numba.guvectorize(
        "(n)->(n)",
        cache=True,
        nopython=True,
    )
    def _localmin(x, y):  # pragma: no cover
        """Vectorized wrapper for the localmin stencil"""
        y[:] = _localmin_sten(x)


    def localmin(x: np.ndarray, *, axis: int = 0) -> NDArray[np.bool]:
        """Find local minima in an array

        An element ``x[i]`` is considered a local minimum if the following
        conditions are met:

        - ``x[i] < x[i-1]``
        - ``x[i] <= x[i+1]``

        Note that the first condition is strict, and that the first element
        ``x[0]`` will never be considered as a local minimum.

        Parameters
        ----------
        x : np.ndarray [shape=(d1,d2,...)]
            input vector or array
        axis : int
            axis along which to compute local minimality

        Returns
        -------
        m : np.ndarray [shape=x.shape, dtype=bool]
            indicator array of local minimality along ``axis``
        """
        # Rotate the target axis to the end
        xi = x.swapaxes(-1, axis)

        # Allocate the output array and rotate target axis
        lmin = np.empty_like(x, dtype=bool)
        lmini = lmin.swapaxes(-1, axis)

        # Call the vectorized stencil
        _localmin(xi, lmini)

        # Handle the edge condition not covered by the stencil
        lmini[..., -1] = xi[..., -1] < xi[..., -2]

        return lmin
