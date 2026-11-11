import numpy as np
import pytest
import scipy.signal
from scipy.ndimage import uniform_filter1d

from iantirta.audio import signal


@pytest.mark.parametrize(
    ('arr, size'),
    [
        # Basic
        ([1, 2, 3, 4, 5, 6], 3),
        ([2, 8, 0, 4, 1, 9, 9, 0], 3),

        # Different odd window sizes
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], 1),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], 3),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], 5),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], 7),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], 9),

        # Short arrays
        ([1], 1),
        ([1], 3),
        ([1, 2], 3),
        ([1, 2], 5),
        ([1, 2, 3], 5),

        # Constant values
        ([5, 5, 5, 5, 5], 3),
        ([5, 5, 5, 5, 5], 5),

        # Zeros
        ([0, 0, 0, 0, 0], 3),

        # Negative values
        ([-5, -2, 0, 3, 7], 3),

        # Mixed negative/positive
        ([-10, 0, 10, -5, 5], 3),

        # Floating point
        ([0.1, 0.2, 0.3, 0.4, 0.5], 3),
        ([1.5, 2.75, -0.5, 4.25, 8.0], 5),

        # Large values
        ([1e-10, 1e10, 1e-5, 1e5, 1.0], 3),
    ]
)
def test_uniform_1d(arr, size):

    arr = np.asarray(arr, dtype=np.float64)

    actual = signal.uniform_1d(arr, size)
    expected = uniform_filter1d(arr, size)

    np.testing.assert_allclose(
        actual,
        expected,
        err_msg=f"Input: {arr}"
    )


@pytest.mark.parametrize(
    ("size", "expected_size"),
    [
        (1, 1),
        (2, 3),
        (3, 3),
        (4, 5),
        (5, 5),
        (6, 7),
    ],
)
def test_uniform_1d_forces_odd_window(size, expected_size):

    arr = np.arange(10, dtype=np.float64)

    actual = signal.uniform_1d(arr, size)
    expected = uniform_filter1d(arr, expected_size)

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("x", "prominence"),
    [
        # No peaks
        ([1, 2, 3, 4, 5], 0.1),
        ([5, 4, 3, 2, 1], 0.1),
        ([1, 1, 1, 1, 1], 0.1),

        # Single peak
        ([0, 5, 0], 0.1),
        ([0, 5, 1], 0.1),
        ([1, 5, 0], 0.1),

        # Peaks at boundaries must not count
        ([5, 0, 1], 0.1),
        ([1, 0, 5], 0.1),
        ([5, 0, 0, 5], 0.1),

        # Multiple peaks
        ([0, 3, 0, 4, 0], 0.1),
        ([0, 1, 0, 2, 0, 3, 0], 0.1),

        # Plateau
        ([0, 1, 1, 0], 0.1),
        ([0, 1, 1, 1, 0], 0.1),
        ([0, 1, 1, 1, 1, 0], 0.1),

        # Plateau with surrounding values
        ([0, 2, 2, 1, 0], 0.1),
        ([0, 1, 2, 2, 2, 1, 0], 0.1),

        # Small prominence should keep both
        ([0, 5, 4, 5, 0], 0.5),

        # Larger prominence removes the smaller peak
        ([0, 5, 4, 5, 0], 2.0),

        # Negative values
        ([-5, -1, -5], 0.1),
        ([-5, -1, -2, -5], 0.5),

        # Floating point
        ([0.0, 0.001, 0.0], 0.0005),
        ([0.0, 1.5, 0.0, 2.5, 0.0], 0.5),
    ],
)
def test_find_peaks_1d(x, prominence):
    x = np.asarray(x, dtype=np.float64)

    expected, _ = scipy.signal.find_peaks(
        x,
        prominence=prominence,
    )

    actual = signal.find_peaks_1d(
        x,
        prominence=prominence,
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("x", "prominence", "wlen"),
    [
        ([0, 5, 0, 4, 0], 1.0, 3),
        ([0, 5, 0, 4, 0], 2.0, 5),

        ([0, 1, 5, 1, 0, 1, 4, 1, 0], 1.0, 3),
        ([0, 1, 5, 1, 0, 1, 4, 1, 0], 1.0, 5),
        ([0, 1, 5, 1, 0, 1, 4, 1, 0], 1.0, 9),

        # Larger-than-signal window
        ([0, 1, 5, 1, 0], 1.0, 99),

        # Even wlen is rounded by SciPy's implementation
        ([0, 1, 5, 1, 0], 1.0, 4),
        ([0, 1, 5, 1, 0], 1.0, 6),
    ],
)
def test_find_peaks_1d_wlen(x, prominence, wlen):
    x = np.asarray(x, dtype=np.float64)

    expected, _ = scipy.signal.find_peaks(
        x,
        prominence=prominence,
        wlen=wlen,
    )

    actual = signal.find_peaks_1d(
        x,
        prominence=prominence,
        wlen=wlen,
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "x",
    [
        [0, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 0, 2, 0],
        [1, 5, 1, 4, 1],
        [-2, -1, -2, 0, -2],
    ],
)
def test_find_peaks_1d_without_prominence(x):
    x = np.asarray(x, dtype=np.float64)

    expected, _ = scipy.signal.find_peaks(x)
    actual = signal.find_peaks_1d(x)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        ([0, 1, 0], [1]),
        ([0, 1, 0, 2, 0], [1, 3]),

        # Plateau: midpoint rounded down
        ([0, 1, 1, 0], [1]),
        ([0, 1, 1, 1, 0], [2]),
        ([0, 1, 1, 1, 1, 0], [2]),

        # No peak
        ([1, 2, 3], []),
        ([3, 2, 1], []),
        ([1, 1, 1], []),

        # Boundary values aren't peaks
        ([5, 0, 1], []),
        ([1, 0, 5], []),

        # Negative
        ([-5, -1, -5], [1]),
    ],
)
def test_local_maxima_1d(x, expected):
    x = np.asarray(x, dtype=np.float64)

    actual, left, right = signal.peaks._local_maxima_1d(x)

    np.testing.assert_array_equal(actual, expected)


def test_local_maxima_1d_plateau_edges():
    x = np.asarray([0, 1, 1, 1, 0], dtype=np.float64)

    midpoints, left_edges, right_edges = (
        signal.peaks._local_maxima_1d(x)
    )

    np.testing.assert_array_equal(midpoints, [2])
    np.testing.assert_array_equal(left_edges, [1])
    np.testing.assert_array_equal(right_edges, [3])


@pytest.mark.parametrize("wlen", [0, 1, -1])
def test_find_peaks_1d_invalid_wlen(wlen):
    x = np.asarray([0, 1, 0], dtype=np.float64)

    with pytest.raises(ValueError):
        signal.find_peaks_1d(
            x,
            prominence=0.1,
            wlen=wlen,
        )


@pytest.mark.parametrize("seed", range(100))
def test_find_peaks_1d_random(seed):
    rng = np.random.default_rng(seed)

    x = rng.normal(size=rng.integers(3, 200))
    prominence = 10 ** rng.uniform(-2, 0)

    expected, _ = scipy.signal.find_peaks(
        x,
        prominence=prominence,
    )

    actual = signal.find_peaks_1d(
        x,
        prominence=prominence,
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("seed", range(100))
def test_find_peaks_1d_random_wlen(seed):
    rng = np.random.default_rng(seed)

    x = rng.normal(size=rng.integers(3, 200))
    prominence = 10 ** rng.uniform(-2, 0)
    wlen = int(rng.integers(2, 100))

    expected, _ = scipy.signal.find_peaks(
        x,
        prominence=prominence,
        wlen=wlen,
    )

    actual = signal.find_peaks_1d(
        x,
        prominence=prominence,
        wlen=wlen,
    )

    np.testing.assert_array_equal(actual, expected)
