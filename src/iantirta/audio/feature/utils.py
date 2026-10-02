# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

import numpy as np

__all__ = ["times_like"]

def times_like(
    x: np.ndarray,
    *,
    sr: float = 22050,
    hop_length: int = 512,
    n_fft: int | None = None,
    axis: int = -1,
) -> np.ndarray:
    if np.isscalar(x):
        frames = np.arange(x)
    else:
        frames = np.arange(x.shape[axis])
    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)
    samples = (np.asanyarray(frames) * hop_length + offset).astype(int)
    time: np.ndarray = np.asanyarray(samples) / float(sr)
    return time
