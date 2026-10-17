# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

import numpy as np

__all__ = [
    "uniform_1d",
]


def uniform_1d(
    arr: np.ndarray,
    size: int,
) -> np.ndarray:
    """Apply a centered uniform moving average with symmetric reflection."""

    if size <= 0:
        raise ValueError("size must be greater than zero")

    if size % 2 == 0:
        size += 1

    pad = size // 2

    arr_padded = np.pad(arr, (pad, pad), "symmetric")
    # Create a convolution kernel
    kernel: np.ndarray = np.ones(size, dtype=arr.dtype) / size
    # Apply convolution to calculate the moving average
    smoothed_arr = np.convolve(arr_padded, kernel, mode="valid")
    return smoothed_arr.astype(arr.dtype, copy=False)
