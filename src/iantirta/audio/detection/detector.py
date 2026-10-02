from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from iantirta.audio.feature.extraction import rms, spectrogram
from iantirta.audio.utils import audio_utils
import scipy.ndimage.uniform_filter1d
import torch.nn.init
import scipy.signal.find_peaks
import librosa
if TYPE_CHECKING:
    import numpy as np

@dataclass(frozen=True, slots=True)
class AudioDetector:
    """Detect regions of activity in an audio waveform.

    Parameters
    ----------
    sample_rate:
        Sampling rate of the input audio in Hz.

    win_length:
        Analysis window length in milliseconds.

    hop_length:
        Distance between consecutive analysis frames in milliseconds.

    num_mel_bins:
        Number of mel-frequency filters.

    fmin:
        Minimum frequency of the mel filter bank in Hz.

    fmax:
        Maximum frequency of the mel filter bank in Hz.

    mel_floor:
        Minimum value applied to the mel spectrogram before logarithmic
        conversion.

    win_function:
        Name of the analysis window.

    """

    sample_rate: int = 16000
    win_length: float = 64.0
    hop_length: float = 16.0
    num_mel_bins: int = 80
    fmin: float = 80
    fmax: float = 7600
    mel_floor: float = 1e-10
    win_function: str = "hann_window"
    
    def detect(
        self,
        audio: np.ndarray,
    ):
        
        peaks_prominence: float = 0.01
        _max = np.max(np.abs(smoothed_rms))
        norm = (smoothed_rms / _max) if _max > 0 else smoothed_rms
        inverted = -norm
        rms_valleys = audio_utils.find_peaks_1d(inverted, prominence=peaks_prominence)

        mask = rms_mask | mel_mask

        # Mask forward
        max_forward_distance_ms: int = 200 # 200ms
        edges = np.where((
            mask[:-1] == True) & (mask[1:] == False
        ))[0]
        maxframesize = int(max_forward_distance_ms / self.precision_ms)
        for e in edges:
            maxdist = e + maxframesize
            after = rms_valleys[rms_valleys > e]
            if len(after) > 0:
                after = after[0]
                mask[e:min(after, maxdist)] = True

        # Mask Merge
        merge_ms: int = 140
        starts, ends = get_mask_starts_ends(mask)
        mergeframe = ms2frame(merge_ms)
        for s, e in zip(ends[:-1], starts[1:]): # i.e. the end of one True block, the start of the next True block
            dist = e - s
            if dist <= mergeframe:
                mask[s:e] = True

        # Mask Delete
        delete_ms: int = 100
        starts, ends = get_mask_starts_ends(mask)
        deleteframe = ms2frame(delete_ms)
        for s, e in zip(starts, ends):
            dur = e - s
            if dur <= deleteframe:
                mask[s:e] = False

        # Merge lower duration
        mask = mask_merge_lower_duration(mask)
        final_mask = _split_final_mask(mask)

        starts, ends = get_mask_starts_ends(final_mask)
        audiosegments = []
        for s, e in zip(starts, ends):
            audiosegments.append(AudioSegment(
                start=self.times[s],
                end=self.times[e] if e < len(self.times) else self.times[-1]
            ))



def get_mask_starts_ends(self, datamask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diffs = np.diff(np.pad(datamask.astype(int), pad_width=1, constant_values=0))
    starts = np.where(diffs == 1)[0]
    ends   = np.where(diffs == -1)[0] 
    return starts, ends

def ms2frame(self, ms: int) -> int:
    return max(1, int(ms / self.precision_ms)) # 140ms gap

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
                prev_dur = (prev_end - prev_start) if (prev_start is not None and prev_end is not None) else float("inf")
                if next_dur < prev_dur:
                    datamask[e:next_start] = True
                else:
                    datamask[prev_end:s] = True
            else:
                raise RuntimeError("Not sure why this happen")
    return datamask

max_maskblock_ms: int = 10000
min_maskblock_ms: int = 1000
min_silence_ms: int = 80

def _split_final_mask(self, datamask: np.ndarray) -> np.ndarray:
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
                        v for v in self.rms.valleys
                        if (cs+minframe) <= v <= (ce-minframe) # - gapframe?
                    ]
                    candidates = []
                    for v in valleys:
                        if not self.mel.mask[v]:
                            # mel mask is False, find star and end?
                            pastmask = self.mel.mask[cs:v]
                            aftermask = self.mel.mask[v:ce]
                            truemask = np.where(pastmask==True)[0]
                            if len(truemask) > 0: maskstart = cs + truemask[-1] + 1
                            else: maskstart = cs
                            truemask = np.where(aftermask==True)[0]
                            if len(truemask) > 0: maskend = v + truemask[0]
                            else: maskend = ce
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
                            best = max(valid_candidates, key=lambda c: c["gap"])
                            beststart, bestend = best["start"], best["end"]
                            datamask[beststart:bestend] = False
                            new_chunks.extend([(cs, beststart), (bestend, ce)])
                        else:
                            new_chunks.append((cs, ce))
                else:
                    new_chunks.append((cs, ce))
            if chunks == new_chunks: break
            chunks = new_chunks
    return datamask
