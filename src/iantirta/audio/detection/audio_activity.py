
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from iantirta.audio import load_audio
from iantirta.audio.feature import RMS, Mel

__all__ = [
    "detect_audio_activiy"
]


def get_mask_starts_ends(datamask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    print("getting mask start and ends", datamask.shape)
    diffs = np.diff(np.pad(
        datamask.astype(int),
        pad_width=1,
        constant_values=0
    ))
    starts = np.where(diffs == 1)[0]
    ends   = np.where(diffs == -1)[0]
    print('result is', starts, ends)
    return starts, ends


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


def mask_uniform(
    arr: np.ndarray,
    dist_ms: int = 140,
):
    """ Any distance `lower than or equal` the threshold ms, would be merged """
    starts, ends = get_mask_starts_ends(arr)
    mergeframe = ms2frame(dist_ms)
    for s, e in zip(ends[:-1], starts[1:]): # i.e. the end of one True block, the start of the next True block
        dist = e - s
        if dist <= mergeframe:
            arr[s:e] = True

    return arr


def mask_delete(
    arr: np.ndarray,
    delete_ms: int = 100
) -> np.ndarray:
    """ Any mask length `lower than or equal` the threshold ms, would be deleted """
    starts, ends = get_mask_starts_ends(arr)
    deleteframe = ms2frame(delete_ms)
    for s, e in zip(starts, ends):
        dur = e - s
        if dur <= deleteframe:
            arr[s:e] = False

    return arr


def mask_merge_lower_duration(
    arr: np.ndarray,
    merge_ms: int = 3000
) -> np.ndarray:
    """ Any mask frame `lower than or equal` the threshold ms, would be merged to the nearest mask """

    while True:
        starts, ends = get_mask_starts_ends(arr)
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
                arr[prev_end:s] = True

            elif next_gap < prev_gap:
                arr[e:next_start] = True

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
                    arr[e:next_start] = True
                else:
                    arr[prev_end:s] = True
            else:
                raise RuntimeError("Not sure why this happen")

    return arr


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


def ms2frame(
    ms: int,
    precision_ms: int = 20,
) -> int:
    return max(1, int(ms / precision_ms))


class AudioSegment:
    pass


@dataclass
class AcvitiyDetectionMetadata:

    audio: str | Path | np.ndarray
    sr: int

    precision_ms: int = 20

    _mel: Mel | None = None
    _rms: RMS | None = None
    _final_mask: np.ndarray | None = None
    _segments: list[AudioSegment] | None = None

    @property
    def hoplength(self) -> int:
        return int((self.sr / 1000) * self.precision_ms)

    @property
    def framelength(self) -> int:
        return round(self.hoplength / (1 - 0.75))

    @property
    def rms(self) -> RMS:
        if self._rms is None:
            self._rms = RMS(self.audio, sr=self.sr, framelength=self.framelength, hoplength=self.hoplength)

        return self._rms

    @property
    def mel(self) -> Mel:
        if self._mel is None:
            self._mel = Mel(self.audio, sr=self.sr, framelength=self.framelength, hoplength=self.hoplength)

        return self._mel

    @property
    def times(self) -> np.ndarray:
        pass

    @property
    def final_mask(self) -> np.ndarray:
        if self._final_mask is None:
            mask = self.rms.mask | self.mel.mask

            mask = mask_merge_forward(mask, self.rms.valleys)
            mask = mask_uniform(mask)
            mask = mask_delete(mask)
            # mask = mask_merge_lower_duration(mask)

            self._final_mask = split_long_block_mask(mask, target_valleys=self.rms.valleys, mel_mask=self.mel.mask)

        return self._final_mask

    @property
    def segments(self) -> list[AudioSegment]:
        if self._segments is None:
            starts, ends = get_mask_starts_ends(self.final_mask)

            segments = []
            for s, e in zip(starts, ends):
                segments.append(AudioSegment(
                    start=self.times[s],
                    end=self.times[e] if e < len(self.times) else self.times[-1]
                ))
            self._segments = segments

        return self._segments


def detect_audio_activiy(
    audio: str | Path | np.ndarray,
    sr: int = 16000,
) -> AcvitiyDetectionMetadata:
    import scipy.signal
    wav, sr = load_audio(audio, sampling_rate=sr, num_channels=1)

    sos = scipy.signal.butter(10, [200, 5000], btype='bandpass', fs=sr, output='sos')
    wav = scipy.signal.sosfilt(sos, wav).astype(np.float32)

    return AcvitiyDetectionMetadata(wav, sr=sr)

if __name__ == '__main__':
    from pathlib import Path
    from iantirta.audio import load_audio
    from rich import inspect as i
    audio_path = str(Path("D:/IANTIRTACOM/dataset/vocals.wav").expanduser().resolve())
    # metadata = detect_audio_activiy(audio_path)
    # i(metadata)
    # print(metadata.final_mask)
    from IPython.display import display, Audio
    wav, sr = load_audio(audio_path)
    display(Audio(wav, rate=sr))
    from huggingface_hub import upload_file