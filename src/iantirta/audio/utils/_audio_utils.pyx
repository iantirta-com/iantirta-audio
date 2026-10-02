import numpy as np
cimport numpy as np

def _local_maxima_1d(const np.float64_t[::1] x not None):
    """
    Find local maxima in a 1D array.

    This function finds all local maxima in a 1D array and returns the indices
    for their edges and midpoints (rounded down for even plateau sizes).

    Parameters
    ----------
    x : ndarray
        The array to search for local maxima.

    Returns
    -------
    midpoints : ndarray
        Indices of midpoints of local maxima in `x`.
    left_edges : ndarray
        Indices of edges to the left of local maxima in `x`.
    right_edges : ndarray
        Indices of edges to the right of local maxima in `x`.

    Notes
    -----
    - Compared to `argrelmax` this function is significantly faster and can
      detect maxima that are more than one sample wide. However this comes at
      the cost of being only applicable to 1D arrays.
    - A maxima is defined as one or more samples of equal value that are
      surrounded on both sides by at least one smaller sample.

    .. versionadded:: 1.1.0
    """
    cdef:
        np.intp_t[::1] midpoints, left_edges, right_edges
        np.intp_t m, i, i_ahead, i_max

    # Preallocate, there can't be more maxima than half the size of `x`
    midpoints = np.empty(x.shape[0] // 2, dtype=np.intp)
    left_edges = np.empty(x.shape[0] // 2, dtype=np.intp)
    right_edges = np.empty(x.shape[0] // 2, dtype=np.intp)
    m = 0  # Pointer to the end of valid area in allocated arrays

    with nogil:
        i = 1  # Pointer to current sample, first one can't be maxima
        i_max = x.shape[0] - 1  # Last sample can't be maxima
        while i < i_max:
            # Test if previous sample is smaller
            if x[i - 1] < x[i]:
                i_ahead = i + 1  # Index to look ahead of current sample

                # Find next sample that is unequal to x[i]
                while i_ahead < i_max and x[i_ahead] == x[i]:
                    i_ahead += 1

                # Maxima is found if next unequal sample is smaller than x[i]
                if x[i_ahead] < x[i]:
                    left_edges[m] = i
                    right_edges[m] = i_ahead - 1
                    midpoints[m] = (left_edges[m] + right_edges[m]) // 2
                    m += 1
                    # Skip samples that can't be maximum
                    i = i_ahead
            i += 1

    # Keep only valid part of array memory.
    midpoints.base.resize(m, refcheck=False)
    left_edges.base.resize(m, refcheck=False)
    right_edges.base.resize(m, refcheck=False)

    return midpoints.base, left_edges.base, right_edges.base


def _peak_prominences(const np.float64_t[::1] x not None,
                      const np.intp_t[::1] peaks not None,
                      np.intp_t wlen):
    """
    Calculate the prominence of each peak in a signal.

    Parameters
    ----------
    x : ndarray
        A signal with peaks.
    peaks : ndarray
        Indices of peaks in `x`.
    wlen : np.intp
        A window length in samples (see `peak_prominences`) which is rounded up
        to the nearest odd integer. If smaller than 2 the entire signal `x` is
        used.

    Returns
    -------
    prominences : ndarray
        The calculated prominences for each peak in `peaks`.
    left_bases, right_bases : ndarray
        The peaks' bases as indices in `x` to the left and right of each peak.

    Raises
    ------
    ValueError
        If a value in `peaks` is an invalid index for `x`.

    Warns
    -----
    PeakPropertyWarning
        If a prominence of 0 was calculated for any peak.

    Notes
    -----
    This is the inner function to `peak_prominences`.

    .. versionadded:: 1.1.0
    """
    cdef:
        np.float64_t[::1] prominences
        np.intp_t[::1] left_bases, right_bases
        np.float64_t left_min, right_min
        np.intp_t peak_nr, peak, i_min, i_max, i
        np.uint8_t show_warning

    show_warning = False
    prominences = np.empty(peaks.shape[0], dtype=np.float64)
    left_bases = np.empty(peaks.shape[0], dtype=np.intp)
    right_bases = np.empty(peaks.shape[0], dtype=np.intp)

    with nogil:
        for peak_nr in range(peaks.shape[0]):
            peak = peaks[peak_nr]
            i_min = 0
            i_max = x.shape[0] - 1
            if not i_min <= peak <= i_max:
                with gil:
                    raise ValueError("peak {} is not a valid index for `x`"
                                     .format(peak))

            if 2 <= wlen:
                # Adjust window around the evaluated peak (within bounds);
                # if wlen is even the resulting window length is implicitly
                # rounded to next odd integer
                i_min = max(peak - wlen // 2, i_min)
                i_max = min(peak + wlen // 2, i_max)

            # Find the left base in interval [i_min, peak]
            i = left_bases[peak_nr] = peak
            left_min = x[peak]
            while i_min <= i and x[i] <= x[peak]:
                if x[i] < left_min:
                    left_min = x[i]
                    left_bases[peak_nr] = i
                i -= 1

            # Find the right base in interval [peak, i_max]
            i = right_bases[peak_nr] = peak
            right_min = x[peak]
            while i <= i_max and x[i] <= x[peak]:
                if x[i] < right_min:
                    right_min = x[i]
                    right_bases[peak_nr] = i
                i += 1

            prominences[peak_nr] = x[peak] - max(left_min, right_min)
            if prominences[peak_nr] == 0:
                show_warning = True

    if show_warning:
        warnings.warn("some peaks have a prominence of 0",
                      PeakPropertyWarning, stacklevel=2)
    # Return memoryviews as ndarrays
    return prominences.base, left_bases.base, right_bases.base
