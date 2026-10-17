# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

import numpy as np

__all__ = [
    "find_peaks_1d",
]


def _local_maxima_1d(x: np.ndarray):
    n = x.size

    # Preallocate, there can't be more maxima than half the size of `x`
    midpoints = np.empty(n // 2, dtype=np.intp)
    left_edges = np.empty(n // 2, dtype=np.intp)
    right_edges = np.empty(n // 2, dtype=np.intp)

    m = 0  # Pointer to the end of valid area in allocated arrays
    i = 1
    i_max = n - 1

    while i < i_max:
        if x[i - 1] < x[i]:
            i_ahead = i + 1
            while i_ahead < i_max and x[i_ahead] == x[i]:
                i_ahead += 1

            if x[i_ahead] < x[i]:
                left_edges[m] = i
                right_edges[m] = i_ahead - 1
                midpoints[m] = (left_edges[m] + right_edges[m]) // 2

                m += 1
                i = i_ahead
        
        i += 1

    return (
        midpoints[:m],
        left_edges[:m],
        right_edges[:m],
    )


def _peak_prominences(
    x: np.ndarray,
    peaks: np.ndarray,
    wlen: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = x.size

    prominences = np.empty(peaks.size, dtype=np.float64)
    left_bases = np.empty(peaks.size, dtype=np.intp)
    right_bases = np.empty(peaks.size, dtype=np.intp)

    for peak_nr, peak in enumerate(peaks):
        i_min = 0
        i_max = n - 1
        if not i_min <= peak <= i_max:
            raise ValueError(
                f"peak {peak} is not a valid index for `x`"
            )

        if 2 <= wlen:
            i_min = max(peak - wlen // 2, i_min)
            i_max = min(peak + wlen // 2, i_max)

        i = left_bases[peak_nr] = peak
        left_min = x[peak]
        left_base = peak

        while i_min <= i and x[i] <= x[peak]:
            if x[i] < left_min:
                left_min = x[i]
                left_base = i
            i -= 1

        i = peak
        right_min = x[peak]
        right_base = peak

        while i <= i_max and x[i] <= x[peak]:
            if x[i] < right_min:
                right_min = x[i]
                right_base = i
            i += 1

        prominences[peak_nr] = (
            x[peak] - max(left_min, right_min)
        )

        # if prominences[peak_nr] == 0:
        #     show_warning = True

        left_bases[peak_nr] = left_base
        right_bases[peak_nr] = right_base

    return prominences, left_bases, right_bases


def find_peaks_1d(
    x: np.ndarray,
    *,
    prominence: float | None = None,
    wlen: int | None = None,
) -> np.ndarray:
    peaks, _, _ = _local_maxima_1d(x)

    if prominence is None:
        return peaks

    if wlen is None:
        wlen = -1
    elif wlen <= 1:
        raise ValueError(
            f"`wlen` must be larger than 1, was {wlen}"
        )
    else:
        wlen = int(np.ceil(wlen))

    prominences, _, _ = _peak_prominences(
        x,
        peaks,
        wlen,
    )

    return peaks[prominences >= prominence]
