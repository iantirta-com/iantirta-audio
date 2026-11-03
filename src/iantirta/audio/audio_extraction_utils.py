import numpy as np


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
