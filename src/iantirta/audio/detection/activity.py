# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from ..feature.extraction import PadMode


def _uniform_1d(
    arr: np.ndarray,
    *,
    uniform_size_ms: int = 60,
    precision_ms: int = 20,
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
    precision_ms: int = 20,
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
    pass


def mask_merge_forward(
    arr: np.ndarray,
    target: np.ndarray,
    *,
    max_distance_ms: int = 200,
    precision_ms: int = 20,
    
) -> np.ndarray:
    edges = np.where((
        arr[:-1] == True) & (arr[1:] == False
    ))[0]
    max_frame_size = int(
        max_distance_ms / precision_ms
    )
    for e in edges:
        max_dist = e + max_frame_size
        after = target[target > e]
        if len(after) > 0:
            after = after[0]
            arr[e:min(after, max_dist)] = True
    return arr


def ms2frame(
    ms: int,
    precision_ms: int = 20,
) -> int:
    return max(1, int(ms / precision_ms))


def get_mask_starts_ends(datamask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diffs = np.diff(np.pad(
        datamask.astype(int),
        pad_width=1,
        constant_values=0
    ))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0] 
    return starts, ends


def mask_uniform(
    datamask: np.ndarray,
    dist_ms: int = 140,
):
    """ Any distance `lower than or equal` the threshold ms, would be merged """
    starts, ends = get_mask_starts_ends(datamask)
    mergeframe = ms2frame(dist_ms)
    for s, e in zip(ends[:-1], starts[1:]): # i.e. the end of one True block, the start of the next True block
        dist = e - s
        if dist <= mergeframe:
            datamask[s:e] = True
    return datamask


def mask_delete(
    datamask: np.ndarray,
    delete_ms: int = 100
) -> np.ndarray:
    """ Any mask length `lower than or equal` the threshold ms, would be deleted """
    starts, ends = get_mask_starts_ends(datamask)
    deleteframe = ms2frame(delete_ms)
    for s, e in zip(starts, ends):
        dur = e - s
        if dur <= deleteframe:
            datamask[s:e] = False
    return datamask


def mask_merge_lower_duration(
    datamask: np.ndarray,
    merge_ms: int = 3000
) -> np.ndarray:
    """ Any mask frame `lower than or equal` the threshold ms, would be merged to the nearest mask """

    while True:
        starts, ends = get_mask_starts_ends(datamask)
        mergeframe = ms2frame(merge_ms)
        if len(starts) == 0:
            break

        shortsegments = []
        for i, (s, e) in enumerate(zip(starts, ends)):
            dur = e - s
            if dur <= mergeframe:
                shortsegments.append((i, s, e))
        if not shortsegments:
            break

        assert len(starts) > 1
        for i, s, e in shortsegments:
            next_start = starts[i+1] if i < len(starts) - 1 else None
            prev_end = ends[i-1] if i > 0 else None
            prev_gap = (s - prev_end) if prev_end is not None else float("inf")
            next_gap = (next_start - e) if next_start is not None else float("inf")
            if prev_gap < next_gap:
                datamask[prev_end:s] = True

            elif next_gap < prev_gap:
                datamask[e:next_start] = True

            elif next_gap == prev_gap:
                next_end = ends[i+1] if i < len(ends) - 1 else None
                next_dur = (next_end - next_start) if (next_start is not None and next_end is not None) else float("inf")
                prev_start = starts[i-1] if i > 0 else None
                prev_dur = (
                    (prev_end - prev_start)
                    if (
                        prev_start is not None
                        and prev_end is not None
                    ) else float("inf")
                )
                if next_dur < prev_dur:
                    datamask[e:next_start] = True
                else:
                    datamask[prev_end:s] = True
            else:
                raise RuntimeError("Not sure why this happen")
    return datamask


def split_long_block_mask(
    datamask: np.ndarray,
    target_valleys: np.ndarray,
    mel_mask: np.ndarray,
    *,
    max_maskblock_ms: int = 10000,  # 10s
    min_maskblock_ms: int = 1000,  # 1s
    min_silence_ms: int = 80,  # 10ms
) -> np.ndarray:
    """ Split Long block Mask Width """
    maxframe = ms2frame(max_maskblock_ms)
    minframe = ms2frame(min_maskblock_ms)

    minsilenceframe = ms2frame(min_silence_ms)

    starts, ends = get_mask_starts_ends(datamask)
    for start, end in zip(starts, ends):
        chunks = [(start, end)]
        while any((
            (ce-cs) >= maxframe
            for cs, ce in chunks
        )):
            new_chunks = []
            for cs, ce in chunks:

                if (ce-cs) >= maxframe:

                    valleys = [
                        v for v in target_valleys
                        if (cs+minframe) <= v <= (ce-minframe) # - gapframe?
                    ]

                    candidates = []
                    for v in valleys:

                        if not mel_mask[v]:
                            # mel mask is False, find star and end?
                            pastmask = mel_mask[cs:v]
                            aftermask = mel_mask[v:ce]
                            truemask = np.where(pastmask == True)[0]

                            if len(truemask) > 0:
                                maskstart = cs + truemask[-1] + 1
                            else:
                                maskstart = cs

                            truemask = np.where(aftermask == True)[0]

                            if len(truemask) > 0:
                                maskend = v + truemask[0]
                            else:
                                maskend = ce

                            assert maskend >= maskstart, (
                                "Negative value for gap"
                                f"\n  maskstart: {maskstart}"
                                f"\n  maskend: {maskend}"
                            )
                            candidates.append({
                                "start": maskstart,
                                "end": maskend,
                                "gap": maskend - maskstart
                            })
                    if candidates:
                        valid_candidates = [
                            c for c in candidates
                            if c["gap"] >= minsilenceframe
                        ]
                        if valid_candidates:
                            # pick the widest gap
                            best = max(
                                valid_candidates,
                                key=lambda c: c["gap"]
                            )
                            beststart, bestend = best["start"], best["end"]
                            datamask[beststart:bestend] = False
                            new_chunks.extend([(cs, beststart), (bestend, ce)])
                        else:
                            new_chunks.append((cs, ce))
                else:
                    new_chunks.append((cs, ce))
            if chunks == new_chunks:
                break
            chunks = new_chunks
    return datamask


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

    rms = detect_and_uniform(
        feature.rms,
        y=audio,
        frame_length=sample_size,
        hop_length=sample_stride
    )

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
    _max = np.max(np.abs(rms))
    norm = (rms / _max) if _max > 0 else rms
    inverted = -norm
    rms_valleys = signal.find_peaks_1d(
        inverted,
        prominence=peaks_prominence
    )

    final_mask = mask_merge_forward(final_mask, rms_valleys)
    final_mask = mask_uniform(final_mask)
    final_mask = mask_delete(final_mask)
    final_mask = mask_merge_lower_duration(final_mask)
    final_mask = split_long_block_mask(
        final_mask,
        rms_valleys,
        mel_mask,
    )

    starts, ends = get_mask_starts_ends(final_mask)
    times = feature.times_like(rms)

    from .base import AudioSegment

    audiosegments = []
    for s, e in zip(starts, ends):
        audiosegments.append(AudioSegment(
            start=times[s],
            end=times[e] if e < len(times) else times[-1]
        ))

    return audiosegments
